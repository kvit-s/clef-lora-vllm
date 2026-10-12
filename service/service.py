"""Clef decision service: the SystemOne request format (TypeSafe's, as Clef's release implements it) in front of a vLLM server that runs
Qwen3.8-27B with the Clef adapter loaded.

POST /v1/systemone takes {"model", "state", "questions"} (the format of Cloudflare's Clef and its
`systemone` function) and answers {"model", "answers", "usage"}. For each request:

1. The record is tokenized by Cloudflare's own encode_record, with Clef's own tokenizer. A
   prompt longer than --max-length is refused rather than cut, since cutting drops the end of the state.
2. The token ids go to vLLM's /v1/completions under the adapter's model name, with max_tokens 1 and
   vllm_xargs {"clef_capture": KEY}, which tells the clef_capture plugin to write the prompt's final
   hidden states to CAPTURE_DIR/KEY.bin. vLLM may serve the start of the prompt from its prefix cache;
   the file then begins at the first position vLLM computed.
3. The scoring head (joint_head.safetensors, run here on the CPU in fp32) turns those hidden states
   into one probability per allowed option, and the answers are shaped by Cloudflare's
   systemone_answer. For the positions vLLM skipped, it uses the key and value rows this service kept
   from an earlier prompt with the same start (reuse.py). If it no longer has them, it asks vLLM again
   with a cache_salt unique to the request, which makes vLLM compute every position.

With --reuse-memory-mb 0 nothing is kept and every request gets a unique cache_salt, so every
prompt is read in full and scored by Cloudflare's JointSchemaHead unchanged.

GET /health reports whether vLLM answers and which adapter is used.

Pictures and video are not accepted yet; a request containing them is refused with 400.

Cloudflare's code (joint_schema_model.py) is loaded from the release directory (clef_code.py);
fetch_release.py downloads the files the service needs.

usage: service.py --release DIR [--vllm URL] [--adapter NAME] [--capture-dir DIR] [--listen HOST:PORT]
                  [--max-length TOKENS] [--reuse-memory-mb MB] [--threads N]
       service.py --release DIR --reference-dir DIR ...   (tests: hidden states saved by reference.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clef_code  # noqa: E402

if __name__ == "__main__":
    clef_code.use_release()  # before anything imports joint_schema_model

import capture_file  # noqa: E402
import reuse  # noqa: E402
from joint_schema_model import (  # noqa: E402
    QUESTION_TYPES,
    JointSchemaHead,
    collate_records,
    encode_record,
    systemone_answer,
)
from lm_rows import OutputEmbeddingRows  # noqa: E402

log = logging.getLogger("clef")


class RequestError(Exception):
    """A problem with the caller's request: answered with 400."""


class BackboneError(Exception):
    """vLLM or the capture failed: answered with 502, or 504 for a timeout."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


class VllmBackbone:
    """Final hidden states of a prompt, computed by vLLM with the Clef adapter."""

    def __init__(self, url: str, adapter: str, capture_dir: str, timeout: float) -> None:
        self.url, self.adapter, self.capture_dir, self.timeout = url.rstrip("/"), adapter, capture_dir, timeout

    def hidden_states(self, token_ids: list[int], reuse_cache: bool, reuse_limit: int = 0) -> tuple[int, torch.Tensor, dict]:
        """(first computed position, hidden states from there on, timings). Without reuse_cache the
        request gets a unique cache_salt, so vLLM computes every position and the first is 0.
        reuse_limit, when set, asks the clef_capture plugin not to reuse cached positions beyond it
        (honoured when vLLM runs with CLEF_FULL_CACHE_HITS=1; otherwise vLLM's own held-back blocks
        keep the reuse further back)."""
        key = "clef-" + uuid.uuid4().hex
        payload = {"model": self.adapter, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0,
                   "vllm_xargs": {"clef_capture": key}}
        if not reuse_cache:
            payload["cache_salt"] = key
        elif reuse_limit:
            payload["vllm_xargs"]["clef_reuse_limit"] = reuse_limit
        body = json.dumps(payload).encode()
        request = urllib.request.Request(self.url + "/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as e:
            raise BackboneError(f"vLLM answered {e.code}: {e.read()[:500].decode(errors='replace')}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise BackboneError(f"vLLM at {self.url} did not answer: {e}") from e
        prompt_tokens = (reply.get("usage") or {}).get("prompt_tokens")
        if prompt_tokens != len(token_ids):
            raise BackboneError(f"vLLM read {prompt_tokens} prompt tokens, {len(token_ids)} were sent")
        path = os.path.join(self.capture_dir, key + ".bin")
        error_path = os.path.join(self.capture_dir, key + ".err")
        deadline = time.monotonic() + 10.0
        while not os.path.exists(path):
            if os.path.exists(error_path):
                message = Path(error_path).read_text()
                os.remove(error_path)
                raise BackboneError(f"hidden states were not captured: {message}")
            if time.monotonic() > deadline:
                raise BackboneError("vLLM answered but wrote no hidden states; is the clef_capture "
                                    "plugin installed and CAPTURE_DIR shared with the container?", 504)
            time.sleep(0.005)
        try:
            header, hidden = capture_file.read(path, token_ids)
        finally:
            os.remove(path)
        return header["first_position"], hidden, {"backbone_seconds": round(time.perf_counter() - started, 4)}

    def prime(self, token_ids: list[int]) -> None:
        """Send the start of a decision prompt with nothing captured, so that vLLM saves a reusable
        point at its end (vLLM needs --prefix-match-unit for a point inside a cache block)."""
        payload = {"model": self.adapter, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0,
                   "vllm_xargs": {"clef_prime": True}}
        request = urllib.request.Request(self.url + "/v1/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response.read()


class Primer:
    """Sends priming requests in the background, one at a time; a newer one replaces one not yet sent.

    After a decision, the next prompt usually repeats this one up to the end of its state and then
    differs (new events, then new questions). Priming the start of this prompt, up to a margin before
    its first question, makes vLLM save a point there that the next prompt can resume from, so it
    reads only what changed and its questions, instead of everything since the last cache block
    boundary (1,552 tokens for Qwen3.8-27B on the measured server). It runs after the answer is sent."""

    def __init__(self, backbone: "VllmBackbone") -> None:
        self.backbone = backbone
        self.pending: list[int] | None = None
        self.condition = threading.Condition()
        threading.Thread(target=self._run, name="primer", daemon=True).start()

    def submit(self, token_ids: list[int]) -> None:
        with self.condition:
            self.pending = token_ids
            self.condition.notify()

    def _run(self) -> None:
        while True:
            with self.condition:
                while self.pending is None:
                    self.condition.wait()
                token_ids, self.pending = self.pending, None
            started = time.perf_counter()
            try:
                self.backbone.prime(token_ids)
                log.info("primed %d tokens in %.3fs", len(token_ids), time.perf_counter() - started)
            except Exception as e:  # noqa: BLE001 - priming only saves time; a failure costs nothing else
                log.warning("priming failed: %s", e)


class ReferenceBackbone:
    """Hidden states saved by reference.py, found by prompt; lets the service be tested without a GPU."""

    def __init__(self, directory: str) -> None:
        self.by_digest = {}
        for npz in Path(directory).glob("*.npz"):
            with np.load(npz) as data:
                self.by_digest[capture_file.token_ids_digest(data["input_ids"])] = npz

    def hidden_states(self, token_ids: list[int], reuse_cache: bool, reuse_limit: int = 0) -> tuple[int, torch.Tensor, dict]:
        npz = self.by_digest.get(capture_file.token_ids_digest(token_ids))
        if npz is None:
            raise BackboneError("no reference hidden states for this prompt")
        with np.load(npz) as data:
            bits = data["hidden_bf16"]
        return 0, torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16), {"backbone_seconds": 0.0}


class Decider:
    def __init__(self, release: str, backbone, max_length: int, store: reuse.PrefixStore | None,
                 prime_margin: int = 0, prime_unit: int = 16) -> None:
        from transformers import AutoTokenizer
        from safetensors.torch import load_file

        release_path = Path(release)
        self.tokenizer = AutoTokenizer.from_pretrained(release_path)
        head_config = json.loads((release_path / "joint_head_config.json").read_text())
        self.head = JointSchemaHead(**head_config)
        self.head.load_state_dict(load_file(release_path / "joint_head.safetensors"), strict=True)
        self.head = self.head.float().eval()
        self.rows = OutputEmbeddingRows(release_path).as_tensor_lookup(torch.float32)
        self.backbone = backbone
        self.max_length = max_length
        self.store = store
        self.head_lock = threading.Lock()
        self.prime_margin, self.prime_unit = prime_margin, prime_unit
        self.primer = Primer(backbone) if prime_margin > 0 and store is not None and hasattr(backbone, "prime") else None

    @staticmethod
    def validate(request: dict) -> None:
        if not isinstance(request, dict):
            raise RequestError("the body must be a JSON object")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise RequestError("model and state are required")
        if request.get("images") or request.get("videos"):
            raise RequestError("images and videos are not supported by this service yet; send text or JSON state")
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise RequestError("at least one question is required")
        for question_id, question in questions.items():
            if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
                raise RequestError(f"{question_id}: type must be noul, choice, or score")
            criteria = question.get("criteria")
            if question["type"] == "choice" and not (isinstance(criteria, dict) and criteria):
                raise RequestError(f"{question_id}: choice criteria must be a non-empty object")
            if question["type"] == "score" and not (isinstance(criteria, list) and criteria):
                raise RequestError(f"{question_id}: score criteria must be a non-empty list")
            if question["type"] == "noul" and criteria is not None and (
                    not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                raise RequestError(f"{question_id}: noul criteria may only describe true and false")

    def decide(self, request: dict) -> dict:
        self.validate(request)
        started = time.perf_counter()
        try:
            # no length limit here: encode_record would meet one by cutting off the end of the state
            encoded = encode_record(self.tokenizer, request, max_length=1 << 40)
        except ValueError as e:
            raise RequestError(str(e)) from e
        token_ids = list(encoded.input_ids)
        if len(token_ids) > self.max_length:
            raise RequestError(f"the prompt is {len(token_ids)} tokens and the limit is {self.max_length}; "
                               "shorten the state")
        needed = reuse.first_needed_position(encoded)
        first, hidden, timings = self.backbone.hidden_states(token_ids, reuse_cache=self.store is not None,
                                                             reuse_limit=needed)
        logits = None
        while logits is None:
            if hidden.shape[0] != len(token_ids) - first or (first and self.store is None):
                raise BackboneError(f"hidden states cover {hidden.shape[0]} positions from {first}, "
                                    f"the prompt has {len(token_ids)}")
            head_started = time.perf_counter()
            with self.head_lock, torch.inference_mode():
                if self.store is None:
                    batch = collate_records([encoded], self.tokenizer.pad_token_id, torch.device("cpu"))
                    logits = self.head(hidden[None].float(), batch["input_ids"], batch["attention_mask"], [encoded],
                                       self.rows)[0]
                else:
                    # Kept rows are read, used and extended under one lock: extending a kept start may
                    # overwrite the rows of its last few blocks.
                    stored = self.store.get(token_ids, first) if 0 < first <= needed else None
                    if first == 0 or stored is not None:
                        hidden = hidden.float()
                        kv = reuse.memory_kv(self.head, hidden)
                        parts = [stored, kv] if stored is not None else [kv]  # attended to without joining
                        logits = reuse.forward(self.head, hidden, first, parts, torch.tensor(token_ids), encoded,
                                               self.rows)
                        self.store.put(token_ids, first, stored, kv, needed)
                        if self.primer is not None:
                            prime_end = (needed - self.prime_margin) // self.prime_unit * self.prime_unit
                            if prime_end > first:
                                self.primer.submit(token_ids[:prime_end])
            if logits is None:
                log.info("vLLM reused %d positions that are not kept here; reading the prompt again in full", first)
                first, hidden, again = self.backbone.hidden_states(token_ids, reuse_cache=False)
                timings["backbone_seconds"] = round(timings["backbone_seconds"] + again["backbone_seconds"], 4)
                timings["read_twice"] = True
        timings["reused_tokens"] = first
        questions = request["questions"]
        answers = {
            q.question_id: systemone_answer(
                questions[q.question_id], dict(zip(q.option_ids, ql.float().softmax(-1).tolist())))
            for q, ql in zip(encoded.questions, logits)
        }
        timings["head_seconds"] = round(time.perf_counter() - head_started, 4)
        timings["total_seconds"] = round(time.perf_counter() - started, 4)
        return {
            "model": request["model"], "answers": answers,
            "usage": {"input_tokens": len(token_ids), "output_tokens": 0},
            "timings": timings,
        }


def make_handler(decider: Decider, backbone_name: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "clef-decider/0.1"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            status = {"ok": True, "backbone": backbone_name}
            if isinstance(decider.backbone, VllmBackbone):
                status["adapter"] = decider.backbone.adapter
            if decider.store is not None:
                status["reuse"] = decider.store.stats()
                try:
                    with urllib.request.urlopen(decider.backbone.url + "/health", timeout=3):
                        status["vllm"] = "up"
                except Exception as e:  # noqa: BLE001
                    status["vllm"] = f"down: {e}"
                    status["ok"] = False
            self._send(200 if status["ok"] else 503, status)

        def _body(self) -> bytes:
            limit = 16 * 1024 * 1024
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                # Some proxies forward request bodies in chunks, without a length.
                parts, size = [], 0
                while True:
                    try:
                        chunk = int(self.rfile.readline(1024).split(b";", 1)[0].strip(), 16)
                    except ValueError as e:
                        raise RequestError("malformed chunked request body") from e
                    if chunk == 0:
                        while self.rfile.readline(65536) not in (b"\r\n", b"\n", b""):
                            pass  # optional trailer lines
                        return b"".join(parts)
                    size += chunk
                    if size > limit:
                        raise RequestError("request larger than 16 MiB")
                    parts.append(self.rfile.read(chunk))
                    self.rfile.readline(3)  # the line break that ends each chunk
            length = int(self.headers.get("Content-Length") or 0)
            if length > limit:
                raise RequestError("request larger than 16 MiB")
            return self.rfile.read(length)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                request = json.loads(self._body() or b"null")
                response = decider.decide(request)
            except (RequestError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            except BackboneError as e:
                log.warning("backbone: %s", e)
                return self._send(e.status, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                log.exception("decision failed")
                return self._send(500, {"error": f"internal error: {e}"})
            log.info("decided %d questions over %d tokens (%d reused) in %.3fs (backbone %.3fs, head %.3fs)",
                     len(response["answers"]), response["usage"]["input_tokens"], response["timings"]["reused_tokens"],
                     response["timings"]["total_seconds"], response["timings"]["backbone_seconds"],
                     response["timings"]["head_seconds"])
            self._send(200, response)

        def log_message(self, fmt, *args) -> None:  # the per-request line above is enough
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE); see fetch_release.py")
    parser.add_argument("--vllm", default="http://127.0.0.1:8000")
    parser.add_argument("--adapter", default="clef")
    parser.add_argument("--capture-dir", default="/dev/shm/clef-hidden",
                        help="where the clef_capture plugin writes; must be the plugin's CLEF_CAPTURE_DIR as seen from here")
    parser.add_argument("--reference-dir", help="serve hidden states saved by reference.py instead of asking vLLM")
    parser.add_argument("--listen", default="127.0.0.1:8091")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prime-margin", type=int, default=0,
                        help="after each decision, prime vLLM's cache with the prompt up to this many tokens before "
                             "its first question (0: off); needs vLLM's --prefix-match-unit to pay off, which the "
                             "DFlash2 fork refuses with a sliding-window draft model (docs/how-it-works.md)")
    parser.add_argument("--prime-unit", type=int, default=16, help="vLLM's --prefix-match-unit")
    parser.add_argument("--reuse-memory-mb", type=int, default=2048,
                        help="memory for the starts of recent prompts (about 48 KB per token); 0 reads every prompt in full")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)

    if args.reference_dir:
        backbone, name = ReferenceBackbone(args.reference_dir), "reference"
    else:
        backbone, name = VllmBackbone(args.vllm, args.adapter, args.capture_dir, args.timeout), "vllm"
    store = reuse.PrefixStore(args.reuse_memory_mb * 2**20) if args.reuse_memory_mb > 0 else None
    decider = Decider(str(clef_code.release_dir()), backbone, args.max_length, store, args.prime_margin, args.prime_unit)
    host, port = args.listen.rsplit(":", 1)
    server = ThreadingHTTPServer((host, int(port)), make_handler(decider, name))
    log.info("listening on %s, hidden states from %s, %s", args.listen, name,
             f"keeping up to {args.reuse_memory_mb} MB of prompt starts" if store else "every prompt read in full")
    server.serve_forever()


if __name__ == "__main__":
    main()
