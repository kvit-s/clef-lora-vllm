"""A stand-in for vLLM, for testing the decision service's vLLM path without a GPU.

It answers the three things the service and the test scripts use:
- POST /v1/completions with token ids and vllm_xargs {"clef_capture": KEY}: writes the prompt's hidden
  states with the clef_capture plugin's own Capture class, in chunks of 2,048 positions as vLLM would,
  then answers with the prompt's token count. The hidden states are reference.py's for its prompts
  when --reference-hidden names a directory of them; for any other prompt they are made up, but
  causally: each position's values follow from the tokens
  up to it, so two prompts with the same start get the same values there, as from a real model.
  Like vLLM's prefix cache, it remembers the points of every prompt it has read from which a later
  prompt can resume: every boundary of --block positions and, with --match-unit (vLLM's
  --prefix-match-unit), the prompt's last boundary of that many positions. A point is named by the
  chain of token ids up to it and by the request's cache_salt. A new prompt resumes from the furthest
  point it shares with an earlier one (at most its length minus one, and at most vllm_xargs
  clef_reuse_limit), and only the positions from there on are written. Requests with vllm_xargs
  clef_prime only register their points;
- POST /v1/chat/completions, streamed or not: a fixed answer, with usage, for speed_probe.py;
- GET /health.

usage: mock_vllm.py CAPTURE_DIR [--reference-hidden DIR] [--listen 127.0.0.1:18199] [--block 1552]
                    [--match-unit N]
"""

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vllm-plugin"))
import capture_file  # noqa: E402
import clef_capture  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("--reference-hidden", help="hidden states saved by eval/reference.py (*.npz)")
    parser.add_argument("--listen", default="127.0.0.1:18199")
    parser.add_argument("--block", type=int, default=1552,
                        help="prefix-cache block size; vLLM uses 1,552 for Qwen3.8-27B")
    parser.add_argument("--match-unit", type=int, default=0, help="like vLLM's --prefix-match-unit (0: the block size)")
    args = parser.parse_args()
    by_digest = {}
    for npz in (Path(args.reference_hidden).glob("*.npz") if args.reference_hidden else []):
        with np.load(npz) as data:
            by_digest[capture_file.token_ids_digest(data["input_ids"])] = npz
    capture = clef_capture.Capture(args.capture_dir)
    unit = args.match_unit or args.block
    assert args.block % unit == 0, "the block size must be a multiple of --match-unit"
    points = set()  # names of the points a later prompt can resume from

    def point_names(ids, salt):
        """names[i] is the name of the point after (i + 1) * unit positions."""
        names, name = [], (salt or "").encode()
        for start in range(0, len(ids) - len(ids) % unit, unit):
            name = hashlib.sha256(name + np.asarray(ids[start:start + unit], dtype="<i4").tobytes()).digest()
            names.append(name)
        return names

    def register(ids, names):
        for index in range(len(names)):
            if ((index + 1) * unit) % args.block == 0:
                points.add(names[index])
        if names:
            points.add(names[-1])  # the prompt's last unit boundary

    def made_up_hidden(ids, first):
        rows, running = [], b""
        for position, token in enumerate(ids):
            running = hashlib.sha256(running + int(token).to_bytes(4, "little")).digest()
            if position >= first:
                rows.append(np.random.default_rng(int.from_bytes(running[:8], "little")).standard_normal(5120))
        return torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(torch.bfloat16)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._json(200, {}) if self.path == "/health" else self._json(404, {})

        def do_POST(self):  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/completions":
                ids = request["prompt"]
                xargs = request.get("vllm_xargs") or {}
                key = xargs.get("clef_capture")
                names = point_names(ids, request.get("cache_salt"))
                longest = min(len(ids) - 1, xargs.get("clef_reuse_limit") or len(ids)) // unit  # vLLM computes the last token
                first = next((count * unit for count in range(longest, 0, -1) if names[count - 1] in points), 0)
                register(ids, names)
                if key:
                    npz = by_digest.get(capture_file.token_ids_digest(ids))
                    if npz is not None:
                        with np.load(npz) as data:
                            hidden = torch.from_numpy(data["hidden_bf16"].view(np.int16).copy()).view(torch.bfloat16)[first:]
                    else:
                        hidden = made_up_hidden(ids, first)
                    for start in range(first, len(ids), 2048):
                        capture.add(key, key, ids, start, hidden[start - first:start - first + 2048])
                    capture.drop([key])
                return self._json(200, {"choices": [{"text": "x"}], "usage": {"prompt_tokens": len(ids)}})
            if self.path == "/v1/chat/completions":
                words = ["token"] * min(int(request.get("max_tokens", 16)), 64)
                usage = {"prompt_tokens": 10, "completion_tokens": len(words)}
                if not request.get("stream"):
                    return self._json(200, {"choices": [{"message": {"content": " ".join(words)}}], "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in words:
                    chunk = {"choices": [{"delta": {"content": word + " "}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    time.sleep(0.002)
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return None
            return self._json(404, {})

        def log_message(self, *a):
            pass

    host, port = args.listen.rsplit(":", 1)
    print(f"mock vLLM on {args.listen} with {len(by_digest)} reference prompts", flush=True)
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
