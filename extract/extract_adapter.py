"""Rebuild Clef's change to Qwen3.8-27B as a rank-256 LoRA adapter, in the PEFT format vLLM loads.

Everything is read over HTTP range requests from Hugging Face, so neither model is downloaded whole:

1. Read both models' safetensors index files and the header of every shard.
2. Compare every tensor of Clef with the same tensor of the base model by hashing its first 256 KiB.
   Clef has 1,184 tensors; 186 differ, and all of them are large matrices of layers 40-63.
3. For each tensor that differs, download both versions in full (37 GB in all), take the
   difference D = W_clef - W_base, factor it as D ~= B @ A with rank 256 from its top singular
   vectors (A = sqrt(S) V^T, B = U sqrt(S)), round A and B to bf16, and measure how closely
   bf16(W_base + B @ A) matches W_clef. The report compares what the factorisation leaves out with
   the rounding noise already in Clef's own bf16 weights.
4. Write adapter_model.safetensors, adapter_config.json and extraction-report.json to OUT_DIR.

On an 8-core desktop CPU this took 23 minutes, most of it downloading. At most --window tensors are
held in the work directory at once (each up to 356 MB, twice).

--only NAME [NAME ...] extracts just those tensors, writing adapter_model.partial.safetensors and
the report, which is a quick way to check the method or compare with a published adapter.

usage: extract_adapter.py OUT_DIR [--work DIR] [--only TENSOR ...] [--threads 6] [--window 12]
       (set HF_TOKEN if your network needs authenticated Hugging Face requests)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RANK = 256
REPOS = {"clef": "Cloudflare/clef", "base": "Qwen/Qwen3.8-27B"}
REVISIONS = {"clef": "2f3de3dd85f379784083b0814d997ab627200f0c", "base": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
SCAN_BYTES = 256 * 1024


class Hub:
    """Range reads of the two models' safetensors files at the pinned revisions."""

    def __init__(self) -> None:
        self.maps = {key: json.loads(self.get(key, "model.safetensors.index.json"))["weight_map"] for key in REPOS}
        self.headers: dict[tuple[str, str], tuple[int, dict]] = {}

    @staticmethod
    def get(key: str, filename: str, start: int | None = None, end: int | None = None) -> bytes:
        url = f"https://huggingface.co/{REPOS[key]}/resolve/{REVISIONS[key]}/{filename}"
        headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        error = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r:
                    data = r.read()
                if start is None or len(data) == end - start + 1:
                    return data
                error = f"got {len(data)} bytes, expected {end - start + 1}"
            except Exception as e:  # noqa: BLE001 - retried, then reported
                error = e
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"download failed: {key} {filename} {start}-{end}: {error}")

    def header(self, key: str, shard: str) -> tuple[int, dict]:
        if (key, shard) not in self.headers:
            length = struct.unpack("<Q", self.get(key, shard, 0, 7))[0]
            self.headers[(key, shard)] = (8 + length, json.loads(self.get(key, shard, 8, 8 + length - 1)))
        return self.headers[(key, shard)]

    def locate(self, key: str, name: str) -> tuple[str, int, int, dict]:
        """(shard, absolute start, absolute end, metadata) of one tensor."""
        shard = self.maps[key][name]
        base, header = self.header(key, shard)
        meta = header[name]
        start, end = meta["data_offsets"]
        return shard, base + start, base + end, meta

    def prefix_hash(self, key: str, name: str) -> str:
        shard, start, end, _ = self.locate(key, name)
        return hashlib.sha256(self.get(key, shard, start, start + min(end - start, SCAN_BYTES) - 1)).hexdigest()

    def tensor(self, key: str, name: str, work: Path) -> tuple[Path, list[int]]:
        """Download one bf16 tensor to the work directory; returns its path and shape."""
        shard, start, end, meta = self.locate(key, name)
        if meta["dtype"] != "BF16":
            raise ValueError(f"{key} {name} is {meta['dtype']}, expected BF16")
        path = work / f"{name}.{key}.bf16"
        if not (path.exists() and path.stat().st_size == end - start):
            data = self.get(key, shard, start, end - 1)
            path.with_suffix(".part").write_bytes(data)
            os.replace(path.with_suffix(".part"), path)
        return path, meta["shape"]


def scan(hub: Hub, work: Path) -> dict[str, bool]:
    """{tensor name: identical in both models}, by the hash of each tensor's first 256 KiB."""
    cache = work / "scan.json"
    if cache.exists():
        return json.loads(cache.read_text())
    names = sorted(hub.maps["clef"])
    for key in REPOS:
        for shard in sorted({hub.maps[key][n] for n in names}):
            hub.header(key, shard)
    with ThreadPoolExecutor(32) as pool:
        result = dict(pool.map(lambda n: (n, hub.prefix_hash("clef", n) == hub.prefix_hash("base", n)), names))
    cache.write_text(json.dumps(result, indent=0))
    return result


def load(path: Path, shape: list[int]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 (round to nearest even), returning the uint16 bit patterns."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def factor(d: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """A (rank x in), B (out x rank) with B @ A the best rank-`rank` approximation of d."""
    import scipy.linalg

    out_dim, in_dim = d.shape
    d64 = d.astype(np.float64)
    if min(out_dim, in_dim) <= rank:
        u, s, vt = np.linalg.svd(d64, full_matrices=False)
    elif in_dim <= out_dim:
        w, v = scipy.linalg.eigh(d64.T @ d64, subset_by_index=[in_dim - rank, in_dim - 1])
        w, v = w[::-1], v[:, ::-1]
        s = np.sqrt(np.clip(w, 0, None))
        u = (d64 @ v) / s
        vt = v.T
    else:
        w, uu = scipy.linalg.eigh(d64 @ d64.T, subset_by_index=[out_dim - rank, out_dim - 1])
        w, uu = w[::-1], uu[:, ::-1]
        s = np.sqrt(np.clip(w, 0, None))
        vt = (uu.T @ d64) / s[:, None]
        u = uu
    k = min(rank, len(s))
    root = np.sqrt(s[:k])
    a = np.zeros((rank, in_dim), dtype=np.float32)
    b = np.zeros((out_dim, rank), dtype=np.float32)
    a[:k] = (root[:, None] * vt[:k]).astype(np.float32)
    b[:, :k] = (u[:, :k] * root[None, :]).astype(np.float32)
    return a, b


def measure(name: str, wc: np.ndarray, wb: np.ndarray, clef_path: Path, a16: np.ndarray, b16: np.ndarray,
            a: np.ndarray, b: np.ndarray) -> dict:
    d = wc - wb
    approx = from_bf16_bits(b16).reshape(b.shape) @ from_bf16_bits(a16).reshape(a.shape)
    rebuilt = wb + approx
    clef16 = np.fromfile(clef_path, dtype=np.uint16).reshape(wc.shape)
    # The rounding noise already present in Clef's stored weights: bf16 keeps 8 significant bits, so a
    # value with exponent e is stored to within half of 2^(e-7), a uniform error with variance spacing^2/12.
    exponent = (clef16.astype(np.int32) >> 7) & 0xFF
    spacing = np.ldexp(np.float32(1), np.maximum(exponent, 1) - 127 - 7).astype(np.float64)
    noise = float(np.sqrt(np.sum(spacing ** 2) / 12))
    return {"tensor": name, "shape": list(wc.shape),
            "change_vs_weight": float(np.linalg.norm(d) / np.linalg.norm(wb)),
            "residual_of_change": float(np.linalg.norm(d - b @ a) / np.linalg.norm(d)),
            "bf16_exact_share": float(np.mean(to_bf16_bits(rebuilt) == clef16)),
            "leftover_vs_clef_rounding_noise": float(np.linalg.norm(d - approx)) / noise,
            "weight_error": float(np.linalg.norm(rebuilt - wc) / np.linalg.norm(wc))}


def write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, list[int]]]) -> None:
    """Written by hand so the factors stay bf16 without needing torch."""
    offset, entries, order = 0, {}, sorted(tensors)
    for k in order:
        bits, shape = tensors[k]
        entries[k] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + bits.nbytes]}
        offset += bits.nbytes
    entries["__metadata__"] = {"format": "pt"}
    head = json.dumps(entries, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for k in order:
            f.write(tensors[k][0].tobytes())


def peft_key(name: str, which: str) -> str:
    return f"base_model.model.{name[:-len('.weight')]}.lora_{which}.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out")
    parser.add_argument("--work", help="directory for downloaded tensors and the scan (default OUT/work)")
    parser.add_argument("--only", nargs="+", help="extract only these tensors")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--window", type=int, default=12, help="tensors downloaded ahead of the one being factored")
    args = parser.parse_args()
    out = Path(args.out)
    work = Path(args.work) if args.work else out / "work"
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()

    hub = Hub()
    same = scan(hub, work)
    changed = sorted(n for n, identical in same.items() if not identical)
    layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in changed if ".layers." in n})
    outside = [n for n in changed if ".layers." not in n or "visual" in n]
    print(f"{len(same)} tensors in Clef, {len(changed)} differ from the base model, in layers "
          f"{layers[0]}-{layers[-1]}" + (f"; also outside the decoder layers: {outside}" if outside else ""), flush=True)
    if args.only:
        unknown = [n for n in args.only if n not in changed]
        if unknown:
            raise SystemExit(f"not among the tensors that differ: {unknown}")
        changed = list(args.only)

    tensors, report = {}, []
    with ThreadPoolExecutor(args.threads) as pool:
        names, pending = iter(changed), deque()

        def submit_next() -> None:
            name = next(names, None)
            if name is not None:
                pending.append((name, pool.submit(hub.tensor, "clef", name, work), pool.submit(hub.tensor, "base", name, work)))

        for _ in range(args.window):
            submit_next()
        index = 0
        while pending:
            name, clef_future, base_future = pending.popleft()
            submit_next()
            (clef_path, shape), (base_path, base_shape) = clef_future.result(), base_future.result()
            assert shape == base_shape, (name, shape, base_shape)
            wc, wb = load(clef_path, shape), load(base_path, shape)
            a, b = factor(wc - wb, RANK)
            a16, b16 = to_bf16_bits(a), to_bf16_bits(b)
            row = measure(name, wc, wb, clef_path, a16, b16, a, b)
            tensors[peft_key(name, "A")] = (a16, list(a.shape))
            tensors[peft_key(name, "B")] = (b16, list(b.shape))
            report.append(row)
            index += 1
            print(f"[{index}/{len(changed)} {time.time() - started:6.0f}s] {name} change {row['change_vs_weight']:.3f} "
                  f"leftover/noise {row['leftover_vs_clef_rounding_noise']:.3f} weight-error {row['weight_error']:.5f}",
                  flush=True)
            clef_path.unlink()
            base_path.unlink()

    out.mkdir(parents=True, exist_ok=True)
    if args.only:
        write_safetensors(out / "adapter_model.partial.safetensors", tensors)
    else:
        write_safetensors(out / "adapter_model.safetensors", tensors)
        config = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": RANK, "lora_dropout": 0.0,
            "bias": "none", "fan_in_fan_out": False, "use_rslora": False, "use_dora": False,
            "target_modules": sorted({n.split(".")[-2] for n in changed}),
            "layers_to_transform": layers,
            "base_model_name_or_path": REPOS["base"], "revision": REVISIONS["base"],
        }
        (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (out / "extraction-report.json").write_text(json.dumps(report, indent=1))
    large = [r for r in report if min(r["shape"]) > RANK]
    print(f"tensors: {len(report)}; parameters: {sum(t[0].size for t in tensors.values()):,}")
    if large:
        print("leftover / Clef's rounding noise, matrices larger than the rank: "
              f"{min(r['leftover_vs_clef_rounding_noise'] for r in large):.3f} to "
              f"{max(r['leftover_vs_clef_rounding_noise'] for r in large):.3f}")
    print(f"largest weight error: {max(r['weight_error'] for r in report):.5f}")
    print(f"done in {time.time() - started:.0f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
