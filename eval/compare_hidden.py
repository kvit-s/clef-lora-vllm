"""Compare vLLM's final hidden states with the full-precision CPU reference, record by record.

For each reference record (reference.py output), the same token ids are sent to vLLM twice with
hidden-state capture on: once under the Clef adapter's model name and once under the plain served
model. Each result is compared with the reference by relative error (|vllm - ref| / |ref| over the
whole prompt) and by the cosine similarity of each position's vector. If the adapter is applied,
the Clef run is much closer to the reference than the plain run; the remaining distance is what
the server's quantization costs. Results are written to compare-hidden.json in the current directory.

usage: compare_hidden.py VLLM_URL CAPTURE_DIR HIDDEN_DIR --base SERVED_MODEL_NAME --release DIR
                         [--adapter clef] [--limit N]
       HIDDEN_DIR holds the hidden states saved by reference.py (*.npz).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
import clef_code  # noqa: E402

clef_code.use_release()
from service import VllmBackbone  # noqa: E402


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(g, r, dim=-1)
    return {"relative_error": float((g - r).norm() / r.norm()), "cosine_mean": float(cos.mean()),
            "cosine_min": float(cos.min()), "last_position_cosine": float(cos[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm")
    parser.add_argument("capture_dir")
    parser.add_argument("reference")
    parser.add_argument("--adapter", default="clef")
    parser.add_argument("--base", required=True, help="the name vLLM serves the base model under")
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE)")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    clef = VllmBackbone(args.vllm, args.adapter, args.capture_dir, 600)
    base = VllmBackbone(args.vllm, args.base, args.capture_dir, 600)
    rows = []
    for npz in sorted(Path(args.reference).glob("*.npz"))[: args.limit]:
        with np.load(npz) as data:
            ids = data["input_ids"].tolist()
            ref = torch.from_numpy(data["hidden_bf16"].view(np.int16).copy()).view(torch.bfloat16)
        _, got_clef, _ = clef.hidden_states(ids, reuse_cache=False)
        _, got_base, _ = base.hidden_states(ids, reuse_cache=False)
        row = {"record": npz.stem, "tokens": len(ids), "clef": compare(got_clef, ref), "plain": compare(got_base, ref)}
        rows.append(row)
        print(f"{npz.stem:28s} {len(ids):5d} tokens  with adapter: error {row['clef']['relative_error']:.3f} "
              f"cosine {row['clef']['cosine_mean']:.4f}  |  without: error {row['plain']['relative_error']:.3f} "
              f"cosine {row['plain']['cosine_mean']:.4f}", flush=True)
    if rows:
        mean = lambda which, key: sum(r[which][key] for r in rows) / len(rows)  # noqa: E731
        print(f"mean relative error: with adapter {mean('clef', 'relative_error'):.3f}, "
              f"without {mean('plain', 'relative_error'):.3f}")
    json.dump(rows, open("compare-hidden.json", "w"), indent=1)


if __name__ == "__main__":
    main()
