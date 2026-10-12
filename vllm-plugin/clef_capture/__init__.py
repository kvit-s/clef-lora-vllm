"""vLLM plugin that hands the final hidden states of marked prompts to the Clef decision service.

Clef's scoring head reads the backbone's final hidden state at every prompt position. vLLM's API
returns tokens, never hidden states, so this plugin wraps the model runner: when a request has
``vllm_xargs: {"clef_capture": KEY}`` (which reaches the runner as sampling_params.extra_args), the
rows of each forward pass that belong to that request's prompt are copied to host memory. A long
prompt arrives in several chunks; they are placed by position, and once every prompt position has
been seen the whole array is written to ``$CLEF_CAPTURE_DIR/KEY.bin`` (default
/dev/shm/clef-hidden; the plugin creates the directory if it is missing and makes it writable by
everyone, since the decision service, which deletes each file after reading it, may run as another
user).

Both of vLLM's GPU model runners are patched: vllm/v1/worker/gpu_model_runner.py and
vllm/v1/worker/gpu/model_runner.py. Stock vLLM 0.31 uses the second for every model unless
VLLM_USE_V2_MODEL_RUNNER=0; the fork this was first run on (0.27.2.dev0) used it when a DFlash
draft model was configured, and the first otherwise.

Tested: on the GPU with vLLM 0.27.2.dev0+g6e448d0ea (the DFlash2 image for RTX 50-series cards),
second runner. Against stock vLLM 0.31.0, only by reading its source: every attribute used below
exists there with the same meaning.

File layout: the 8 bytes ``CLEFHS1\\n``, a little-endian u64 header length, a JSON header
(key, prompt_tokens, first_position, hidden_size, dtype "bf16", token_ids_sha256 of all the prompt
ids as little-endian int32), then (prompt_tokens - first_position) x hidden_size bf16 values,
row-major, for positions first_position onwards.

first_position is above 0 when vLLM served the start of the prompt from its prefix cache: those
positions were never computed, so they have no hidden states. The decision service fills them in
from what it kept of an earlier prompt with the same start, or asks again with a unique cache_salt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct

logger = logging.getLogger("vllm.clef_capture")

MAGIC = b"CLEFHS1\n"
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


def capture_key(sampling_params) -> str | None:
    extra = getattr(sampling_params, "extra_args", None) or {}
    key = extra.get("clef_capture") if isinstance(extra, dict) else None
    if key is None:
        return None
    if not isinstance(key, str) or not KEY_PATTERN.match(key):
        logger.warning("clef_capture: ignoring malformed key %r", key)
        return None
    return key


class Capture:
    """Per-request host buffers, living in the model runner's process."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._ensure_directory()
        self.buffers: dict[str, dict] = {}

    def _ensure_directory(self) -> None:
        # The directory is in the host's /dev/shm, which is empty after WSL restarts. Whichever
        # creates it first, this plugin (as the container's vllm user) or the decision service (as
        # its own user), makes it writable by both: the service deletes each file after reading it.
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)
            try:
                os.chmod(self.directory, 0o777)
            except PermissionError:
                pass

    def add(self, req_id: str, key: str, prompt_ids, start: int, rows) -> None:
        import numpy as np
        import torch

        entry = self.buffers.get(req_id)
        if entry is None:
            # The first chunk starts where vLLM's prefix cache stopped; positions before it were not computed.
            length = len(prompt_ids)
            entry = {
                "key": key, "done": False, "length": length, "first": start, "prompt_ids": list(prompt_ids),
                "array": np.empty((length - start, rows.shape[1]), dtype=np.uint16),
                "seen": np.zeros(length - start, dtype=bool),
            }
            self.buffers[req_id] = entry
        if entry["done"]:
            return
        if start < entry["first"]:
            self._error(key, f"chunk at position {start} is before the first computed position {entry['first']}")
            entry["done"] = True
            return
        n = rows.shape[0]
        host = rows.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        at = start - entry["first"]
        entry["array"][at:at + n] = host
        entry["seen"][at:at + n] = True
        if entry["seen"].all():
            self._write(entry)
            entry["done"] = True

    def fail(self, req_id: str, key: str, message: str) -> None:
        """Report a request that cannot be captured, once."""
        entry = self.buffers.setdefault(req_id, {"key": key, "done": False})
        if not entry["done"]:
            self._error(key, message)
            entry["done"] = True

    def drop(self, req_ids) -> None:
        for req_id in req_ids:
            self.buffers.pop(req_id, None)

    def _write(self, entry: dict) -> None:
        import numpy as np

        ids = np.asarray(entry["prompt_ids"], dtype="<i4").tobytes()
        header = json.dumps({
            "key": entry["key"], "prompt_tokens": entry["length"], "first_position": entry["first"],
            "hidden_size": int(entry["array"].shape[1]),
            "dtype": "bf16", "token_ids_sha256": hashlib.sha256(ids).hexdigest(),
        }).encode()
        self._ensure_directory()
        final = os.path.join(self.directory, entry["key"] + ".bin")
        part = final + ".part"
        with open(part, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(entry["array"].tobytes())
        os.replace(part, final)

    def _error(self, key: str, message: str) -> None:
        self._ensure_directory()
        final = os.path.join(self.directory, key + ".err")
        with open(final + ".part", "w") as f:
            f.write(message)
        os.replace(final + ".part", final)
        logger.warning("clef_capture %s: %s", key, message)


def _directory() -> str:
    return os.environ.get("CLEF_CAPTURE_DIR", "/dev/shm/clef-hidden")


def _hidden_of(model_output):
    hidden = model_output[0] if isinstance(model_output, tuple) else model_output
    if not hasattr(hidden, "dim") or hidden.dim() != 2:
        return None  # pipeline-parallel intermediate tensors, or something this plugin does not understand
    return hidden


def _synchronize(hidden) -> None:
    # The forward may have been queued on a stream other than the current one; wait for all of it
    # before reading. This only happens in steps that contain a marked prompt.
    import torch

    if hidden.is_cuda:
        torch.cuda.synchronize(hidden.device)


def _patch_v1() -> None:
    """vllm/v1/worker/gpu_model_runner.py: used when no DFlash2 draft is configured."""
    from vllm.v1.worker import gpu_model_runner

    runner_class = gpu_model_runner.GPUModelRunner
    if getattr(runner_class, "_clef_capture_patched", False):
        return
    original_execute = runner_class.execute_model
    original_forward = runner_class._model_forward

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not hasattr(self, "_clef_capture"):
            self._clef_capture = Capture(_directory())
        self._clef_scheduler_output = scheduler_output
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            self._clef_scheduler_output = None
            self._clef_capture.drop(getattr(scheduler_output, "finished_req_ids", ()) or ())

    def _model_forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        try:
            _capture_v1(self, output)
        except Exception:  # noqa: BLE001 - never let capture break serving
            logger.exception("clef_capture: capture failed")
        return output

    runner_class.execute_model = execute_model
    runner_class._model_forward = _model_forward
    runner_class._clef_capture_patched = True


def _capture_v1(runner, model_output) -> None:
    scheduler_output = getattr(runner, "_clef_scheduler_output", None)
    if scheduler_output is None:
        return  # a dummy or profiling run
    batch = runner.input_batch
    req_ids = batch.req_ids[:batch.num_reqs]
    marked = []
    offset = 0
    for index, req_id in enumerate(req_ids):
        count = scheduler_output.num_scheduled_tokens.get(req_id, 0)
        state = runner.requests.get(req_id)
        key = capture_key(getattr(state, "sampling_params", None)) if state is not None else None
        if key:
            marked.append((req_id, key, state.prompt_token_ids or [], int(batch.num_computed_tokens_cpu[index]),
                           offset, count))
        offset += count
    if not marked:
        return
    hidden = _hidden_of(model_output)
    if hidden is None:
        return
    _synchronize(hidden)
    for req_id, key, prompt_ids, start, offset, count in marked:
        take = max(0, min(count, len(prompt_ids) - start))
        if take:
            runner._clef_capture.add(req_id, key, prompt_ids, start, hidden[offset:offset + take])


def _patch_v2() -> None:
    """vllm/v1/worker/gpu/model_runner.py: forced by a DFlash2 draft model."""
    from vllm.v1.worker.gpu import model_runner

    runner_class = model_runner.GPUModelRunner
    if getattr(runner_class, "_clef_capture_patched", False):
        return
    original_add = runner_class.add_requests
    original_execute = runner_class.execute_model

    def add_requests(self, scheduler_output):
        marked = self.__dict__.setdefault("_clef_marked", {})
        for new in scheduler_output.scheduled_new_reqs:
            key = capture_key(getattr(new, "sampling_params", None))
            if key:
                marked[new.req_id] = (key, list(new.prompt_token_ids or []))
        return original_add(self, scheduler_output)

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not hasattr(self, "_clef_capture"):
            self._clef_capture = Capture(_directory())
        result = original_execute(self, scheduler_output, *args, **kwargs)
        dummy_run = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if not dummy_run:
            try:
                _capture_v2(self)
            except Exception:  # noqa: BLE001 - never let capture break serving
                logger.exception("clef_capture: capture failed")
            finished = getattr(scheduler_output, "finished_req_ids", ()) or ()
            self._clef_capture.drop(finished)
            for req_id in finished:
                self.__dict__.get("_clef_marked", {}).pop(req_id, None)
        return result

    runner_class.add_requests = add_requests
    runner_class.execute_model = execute_model
    runner_class._clef_capture_patched = True


def _capture_v2(runner) -> None:
    marked = runner.__dict__.get("_clef_marked")
    state = getattr(runner, "execute_model_state", None)
    if not marked or state is None or state.hidden_states is None:
        return
    batch = state.input_batch
    work = []
    for index, req_id in enumerate(batch.req_ids[:batch.num_reqs]):
        if req_id in marked:
            key, prompt_ids = marked[req_id]
            work.append((req_id, key, prompt_ids, int(batch.num_computed_tokens_np[index]),
                         int(batch.query_start_loc_np[index]), int(batch.num_scheduled_tokens[index])))
    if not work:
        return
    if getattr(runner, "adaptive_verification", None) is not None:
        # vLLM 0.31's adaptive draft verification moves requests' rows on the GPU after the CPU-side
        # offsets this plugin reads are computed, so the rows cannot be placed reliably.
        for req_id, key, *_ in work:
            runner._clef_capture.fail(req_id, key, "hidden-state capture does not support adaptive verification")
        return
    hidden = _hidden_of(state.hidden_states)
    if hidden is None:
        return
    _synchronize(hidden)
    for req_id, key, prompt_ids, start, offset, count in work:
        take = max(0, min(count, len(prompt_ids) - start))
        if take:
            runner._clef_capture.add(req_id, key, prompt_ids, start, hidden[offset:offset + take])


LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_range(spec: str) -> range:
    first, _, last = spec.partition("-")
    return range(int(first), int(last or first) + 1)


def _patch_lora_layers() -> bool:
    """Give LoRA buffers only to decoder layers in $CLEF_LORA_LAYERS (e.g. "40-63").

    vLLM allocates adapter buffers of the full --max-lora-rank for every module whose name matches,
    in every layer. Clef changes only layers 40-63, so without this the other 40 layers would hold
    about 2.3 GB of graphics memory that is never used. Vision-tower modules are excluded too.

    Which function decides differs between vLLM versions: in 0.27 every module goes through
    LoRAModelManager._match_target_modules; in 0.31 that is used only with --lora-target-modules,
    and otherwise is_supported_lora_module (called by name inside vllm.lora.model_manager) decides.
    Both are wrapped. The log line says the limit is on in either case, so check the saving by the
    graphics memory vLLM reports, or by the KV cache size it fits in the same memory.
    """
    spec = os.environ.get("CLEF_LORA_LAYERS")
    if not spec:
        return False
    from vllm.lora import model_manager

    manager_class = model_manager.LoRAModelManager
    if getattr(manager_class, "_clef_layers_patched", False):
        return True
    allowed = _layer_range(spec)

    def in_allowed_layer(module_name: str) -> bool:
        if "visual" in module_name:
            return False
        match = LAYER_PATTERN.search(module_name)
        return match is not None and int(match.group(1)) in allowed

    original_match = manager_class._match_target_modules

    def _match_target_modules(self, module_name: str) -> bool:
        return original_match(self, module_name) and in_allowed_layer(module_name)

    manager_class._match_target_modules = _match_target_modules
    wrapped = ["LoRAModelManager._match_target_modules"]
    original_supported = getattr(model_manager, "is_supported_lora_module", None)
    if original_supported is not None:
        def is_supported_lora_module(module_name, *args, **kwargs):  # 2 arguments in 0.27, 3 in 0.31
            return original_supported(module_name, *args, **kwargs) and in_allowed_layer(module_name)

        model_manager.is_supported_lora_module = is_supported_lora_module
        wrapped.append("model_manager.is_supported_lora_module")
    manager_class._clef_layers_patched = True
    logger.info("clef_capture: LoRA buffers limited to decoder layers %s (wrapped %s)", spec, ", ".join(wrapped))
    return True


def _patch_decision_cache_hits() -> bool:
    """Let decision requests reuse every whole cached block of their prompt ($CLEF_FULL_CACHE_HITS=1).

    With speculative decoding, vLLM holds back the last matching block of every prefix-cache hit
    (the "EAGLE last-block drop" in vllm/v1/core/kv_cache_coordinator.py), because EAGLE-style draft
    models compute a block's draft cache from the token after it. The DFlash2 release also holds back
    one more block of the linear-attention state, which may have been written over rejected draft
    tokens while generating. A decision (max_tokens 1) uses no draft tokens, and its prompt's states
    are all written while reading prompts, so both are skipped for requests marked with clef_capture:
    on the measured server that is 3,104 tokens fewer to read per reused decision. The scheduler is
    also made to save the linear-attention state at a decision prompt's last block boundary (below).

    When the request also has vllm_xargs {"clef_reuse_limit": N}, the hit is capped at N tokens,
    so that an identical prompt sent again is not served from the cache past the start of its first
    question, whose hidden states the scoring network reads directly.

    Written against vLLM 0.27.2.dev0 (KVCacheManager.get_computed_blocks and the hybrid coordinator's
    attention_groups); a different coordinator is left alone.
    """
    if os.environ.get("CLEF_FULL_CACHE_HITS") != "1":
        return False
    from vllm.v1.core import kv_cache_manager

    manager_class = kv_cache_manager.KVCacheManager
    if getattr(manager_class, "_clef_hits_patched", False):
        return True
    original = manager_class.get_computed_blocks

    def get_computed_blocks(self, request):
        if not _is_decision(request):
            return original(self, request)
        params = request.sampling_params
        coordinator = getattr(self, "coordinator", None)
        groups = getattr(coordinator, "attention_groups", None)
        if groups is None:
            return original(self, request)
        limit = (params.extra_args or {}).get("clef_reuse_limit")
        find = coordinator.find_longest_cache_hit
        # The scheduler runs in one thread, so changing the coordinator for the length of this call
        # affects this request only.
        coordinator.attention_groups = type(groups)(group._replace(use_eagle=False) for group in groups)
        if isinstance(limit, int) and limit > 0:
            coordinator.find_longest_cache_hit = (
                lambda block_hashes, max_length, *args, **kwargs: find(block_hashes, min(max_length, limit), *args, **kwargs))
        try:
            return original(self, request)
        finally:
            coordinator.attention_groups = groups
            coordinator.__dict__.pop("find_longest_cache_hit", None)

    manager_class.get_computed_blocks = get_computed_blocks
    manager_class._clef_hits_patched = True

    # The scheduler cuts a prompt into chunks so that the linear-attention state is saved at block
    # boundaries, but with speculative decoding it stops one block short of the prompt's last
    # boundary ("FullAttn prunes the last matching block, so back off one block",
    # Scheduler._mamba_block_aligned_split), so the state there is never saved. A decision whose
    # hits are not cut short wants that state, for the next, longer prompt to reuse.
    from vllm.v1.core.sched import scheduler as scheduler_module

    scheduler_class = scheduler_module.Scheduler
    original_split = getattr(scheduler_class, "_mamba_block_aligned_split", None)
    if original_split is not None:
        def _mamba_block_aligned_split(self, request, *args, **kwargs):
            if not getattr(self, "use_eagle", False) or not _is_decision(request):
                return original_split(self, request, *args, **kwargs)
            self.use_eagle = False
            try:
                return original_split(self, request, *args, **kwargs)
            finally:
                self.use_eagle = True

        scheduler_class._mamba_block_aligned_split = _mamba_block_aligned_split
    logger.info("clef_capture: decision requests reuse every whole cached block (no held-back blocks)%s",
                "" if original_split is not None else "; scheduler chunking not patched")
    return True


def _is_decision(request) -> bool:
    """A one-token request that this plugin captures, or that primes the cache for decisions.

    A priming request (vllm_xargs {"clef_prime": true}) is the start of a decision prompt, sent by the
    decision service after answering, so that vLLM saves a reusable point (with --prefix-match-unit,
    the prompt's end) just before where the next, longer prompt departs. It needs the same cache
    treatment as a decision and no capture."""
    if getattr(request, "max_tokens", None) != 1:
        return False
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    return capture_key(params) is not None or (isinstance(extra, dict) and extra.get("clef_prime") is True)


TESTED_VLLM = ("0.27.2.dev0", "0.31.0")


def register() -> None:
    """vLLM general-plugin entry point; runs in every vLLM process."""
    import vllm

    version = getattr(vllm, "__version__", "unknown")
    if not version.startswith(TESTED_VLLM):
        logger.warning("clef_capture: vLLM %s; this plugin was run on 0.27.2.dev0 and checked against the "
                       "source of 0.31.0. Run eval/compare_hidden.py before relying on it.", version)
    patched = []
    for name, patch in (("v1", _patch_v1), ("v2", _patch_v2), ("lora-layers", _patch_lora_layers),
                        ("cache-hits", _patch_decision_cache_hits)):
        try:
            if patch() is not False:
                patched.append(name)
        except ImportError:
            logger.debug("clef_capture: %s not present in this process", name, exc_info=True)
        except Exception:  # noqa: BLE001 - report, but never stop vLLM from starting
            logger.warning("clef_capture: could not patch %s; this vLLM version differs from the tested ones",
                           name, exc_info=True)
    if patched:
        logger.info("clef_capture: patched %s; hidden states go to %s", ",".join(patched), _directory())
