"""Clef's scoring network inside vLLM, on the GPU ($CLEF_GPU_HEAD=1).

Instead of writing a decision prompt's hidden states to a file for the decision service to score on
the CPU, the plugin scores the prompt itself, chunk by chunk, as vLLM computes it:

- for each chunk it computes the twelve key and value rows of each position (head.memory_kv); rows of
  positions before the first question go into a store of prompt starts on the GPU, in NVFP4
  (gpu_store.py), and the rows and hidden states of the positions from there on (the questions and
  their options, a few hundred positions) are kept at full precision for this request;
- when every position has been seen, it runs the rest of the scoring network (head.forward) over the
  stored rows and the kept ones, and writes the option scores to CAPTURE_DIR/KEY.json.

The decision service sends, in vllm_xargs "clef_head", where the questions and options lie in the
prompt, as JSON text: {"cut": first question position rounded down to 16, "questions": [{"type": t,
"span": [s, e], "options": [[s, e], ...]}, ...]}. If vLLM served the start of the prompt from its cache and the store
no longer has its rows, the plugin writes KEY.err starting with "not kept", and the service asks again
with a cache_salt, which makes vLLM read the whole prompt.

Configuration: CLEF_RELEASE (Clef's release directory: joint_schema_model.py, joint_head.safetensors,
joint_head_config.json and the output-embedding matrix), CLEF_GPU_STORE_MB (default 512).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import head as clef_head
from .gpu_store import GpuStore

logger = logging.getLogger("vllm.clef_capture")


def load_cloudflare_code(release: Path):
    """Cloudflare's joint_schema_model.py from the release, loaded by path."""
    if "joint_schema_model" in sys.modules:
        return sys.modules["joint_schema_model"]
    spec = importlib.util.spec_from_file_location("joint_schema_model", release / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = module
    spec.loader.exec_module(module)
    return module


class OutputRows:
    """Rows of Clef's output-embedding matrix, read from disk on demand and moved to the device."""

    NAME = "lm_head.weight"

    def __init__(self, release: Path, device: torch.device) -> None:
        shard = release / "lm_head.safetensors"
        if not shard.exists():
            index = json.loads((release / "model.safetensors.index.json").read_text())
            shard = release / index["weight_map"][self.NAME]
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(header_len))[self.NAME]
        start, _ = meta["data_offsets"]
        self.matrix = np.memmap(shard, dtype=np.uint16, mode="r", offset=8 + header_len + start, shape=tuple(meta["shape"]))
        self.device = device

    def __getitem__(self, token_ids: torch.Tensor) -> torch.Tensor:
        bits = np.ascontiguousarray(self.matrix[token_ids.detach().cpu().numpy().astype(np.int64)])
        return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(self.device, torch.float32)


class GpuScorer:
    def __init__(self, release: str, store_bytes: int, directory: str) -> None:
        self.release, self.store_bytes, self.directory = Path(release), store_bytes, directory
        self.head = None
        self.states: dict[str, SimpleNamespace] = {}

    def _load(self, device: torch.device) -> None:
        from safetensors.torch import load_file

        module = load_cloudflare_code(self.release)
        config = json.loads((self.release / "joint_head_config.json").read_text())
        head = module.JointSchemaHead(**config)
        head.load_state_dict(load_file(self.release / "joint_head.safetensors"))
        self.head = head.float().eval().to(device)
        self.rows = OutputRows(self.release, device)
        self.store = GpuStore(self.store_bytes, device, width=config["width"])
        logger.info("clef_capture: scoring network loaded on %s; store of prompt starts %d MB", device,
                    self.store_bytes // 2**20)

    def add(self, request_id: str, key: str, prompt_ids, spec: dict, start: int, hidden: torch.Tensor) -> None:
        """hidden: the last layer's output for prompt positions start..start+n of one request."""
        # The plugin's hook runs after vLLM's execute_model has returned, outside its inference mode.
        with torch.inference_mode():
            self._add(request_id, key, prompt_ids, spec, start, hidden)

    def _add(self, request_id: str, key: str, prompt_ids, spec: dict, start: int, hidden: torch.Tensor) -> None:
        state = self.states.get(request_id)
        try:
            if state is None:
                state = self._begin(request_id, key, prompt_ids, spec, start, hidden.device)
            if state.done:
                return
            if start != state.next:
                raise RuntimeError(f"chunk at position {start}, expected {state.next}")
            self._chunk(state, start, hidden)
            if state.next >= len(state.prompt_ids):
                self._finish(state)
        except Exception as e:  # noqa: BLE001 - never let scoring break serving
            logger.exception("clef_capture: scoring failed")
            if state is not None and not state.done:
                self._fail(state, f"scoring failed: {e}")
            elif state is None:
                self._write(key, ".err", f"scoring failed: {e}")
                self.states[request_id] = SimpleNamespace(done=True, segment=None)

    def _begin(self, request_id, key, prompt_ids, spec, start, device) -> SimpleNamespace:
        if self.head is None:
            self._load(device)
        cut = int(spec["cut"])
        questions = [SimpleNamespace(question_type=int(q["type"]), question_span=tuple(q["span"]),
                                     option_spans=[tuple(o) for o in q["options"]]) for q in spec["questions"]]
        state = SimpleNamespace(key=key, prompt_ids=list(prompt_ids), cut=cut, first=start, next=start, done=False,
                                record=SimpleNamespace(questions=questions), segment=None, write_from=cut,
                                tail_kv=[], tail_hidden=[], started=time.perf_counter(), busy=0.0)
        self.states[request_id] = state
        if not 0 < cut <= clef_head.first_needed_position(state.record) or start > cut:
            raise ValueError(f"cut {cut} and first computed position {start} do not fit the questions")
        reserved = self.store.begin(request_id, state.prompt_ids, start, cut)
        if reserved is None:
            self._fail(state, f"not kept: rows of positions 0..{start} are not in the store")
            return state
        state.segment, state.write_from = reserved
        return state

    def _chunk(self, state: SimpleNamespace, start: int, hidden: torch.Tensor) -> None:
        end = start + hidden.shape[0]
        state.next = end
        lo = max(start, state.write_from)  # rows before write_from are already stored
        if lo >= end:
            return
        began = time.perf_counter()
        h = hidden[lo - start:].float()
        kv = clef_head.memory_kv(self.head, h)
        split = min(max(state.cut, lo), end)
        if split > lo:
            self.store.write(state.segment, lo, kv[:, :split - lo])
        if end > split:
            state.tail_kv.append(kv[:, split - lo:])
            state.tail_hidden.append(h[split - lo:])
        if h.is_cuda:
            torch.cuda.synchronize(h.device)  # so that the time counted is the work done
        state.busy += time.perf_counter() - began

    def _finish(self, state: SimpleNamespace) -> None:
        began = time.perf_counter()
        tail_kv = torch.cat(state.tail_kv, dim=1)
        tail_hidden = torch.cat(state.tail_hidden)
        parts = [self.store.view(state.segment, state.cut), tail_kv]
        ids = torch.tensor(state.prompt_ids)
        logits = clef_head.forward(self.head, tail_hidden, state.cut, parts, ids, state.record, self.rows)
        scores = [row.float().tolist() for row in logits]
        self.store.finish(state.segment, state.cut)
        state.done = True
        state.tail_kv, state.tail_hidden = [], []
        state.busy += time.perf_counter() - began
        self._write(state.key, ".json", json.dumps({
            "logits": scores, "first_position": state.first,
            "scoring_seconds": round(state.busy, 4),            # the plugin's own work, across all chunks
            "since_first_chunk_seconds": round(time.perf_counter() - state.started, 4),
            "store": self.store.stats()}))

    def _fail(self, state: SimpleNamespace, message: str) -> None:
        if state.segment is not None:
            self.store.abort(state.segment)
        state.done, state.tail_kv, state.tail_hidden = True, [], []
        self._write(state.key, ".err", message)

    def drop(self, request_ids) -> None:
        for request_id in request_ids:
            state = self.states.pop(request_id, None)
            if state is not None and not state.done and state.segment is not None:
                self.store.abort(state.segment)

    def _write(self, key: str, suffix: str, text: str) -> None:
        os.makedirs(self.directory, exist_ok=True)
        final = os.path.join(self.directory, key + suffix)
        with open(final + ".part", "w") as f:
            f.write(text)
        os.replace(final + ".part", final)


def scorer_from_environment(directory: str) -> GpuScorer | None:
    if os.environ.get("CLEF_GPU_HEAD") != "1":
        return None
    release = os.environ.get("CLEF_RELEASE")
    if not release or not (Path(release) / "joint_schema_model.py").is_file():
        logger.warning("clef_capture: CLEF_GPU_HEAD=1 but CLEF_RELEASE does not name Clef's release directory")
        return None
    return GpuScorer(release, int(os.environ.get("CLEF_GPU_STORE_MB", "512")) * 2**20, directory)
