"""Download the parts of Cloudflare's Clef release that the decision service needs, about 2.8 GB.

The service never runs Clef's backbone (vLLM does, as the base model plus the adapter), so it does
not need the 52 GB of weights. It needs:

- joint_schema_model.py: Cloudflare's prompt layout, scoring network and answer format;
- joint_head.safetensors and joint_head_config.json: the scoring network's weights (256 MB);
- tokenizer.json and tokenizer_config.json: Clef's tokenizer, which differs from the base model's
  in how it splits text before tokenizing;
- the output-embedding matrix (lm_head.weight, 248,320 x 5,120 bf16, 2.5 GB), because the scoring
  network averages the rows of each option's tokens. It is identical in Clef and the base model.
  Only that tensor is downloaded, by HTTP range requests on the shard that holds it, and written as
  lm_head.safetensors; lm_rows.py reads it from there.

The small files of the release (config, license, model card, index) are downloaded too, so the
directory looks like a partial copy of the release. Everything comes from the revision this
repository was tested with. A file already present with the right size is not downloaded again.

usage: fetch_release.py OUT_DIR
       (set HF_TOKEN if your network needs authenticated Hugging Face requests)
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_code import CLEF_REPO, CLEF_REVISION  # noqa: E402

SMALL_FILES = [
    "LICENSE", "README.md", "chat_template.jinja", "config.json", "generation_config.json",
    "joint_head_config.json", "joint_schema_model.py", "model.safetensors.index.json",
    "processor_config.json", "tokenizer.json", "tokenizer_config.json",
]
HEAD_FILE = "joint_head.safetensors"
LM_HEAD = "lm_head.weight"
CHUNK = 64 * 2**20


def url(name: str) -> str:
    return f"https://huggingface.co/{CLEF_REPO}/resolve/{CLEF_REVISION}/{name}"


def fetch(name: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    error = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url(name), headers=headers), timeout=600) as response:
                data = response.read()
            if start is None or len(data) == end - start + 1:
                return data
            error = f"got {len(data)} bytes, expected {end - start + 1}"
        except Exception as e:  # noqa: BLE001 - retried, then reported
            error = e
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not download {name}: {error}")


def save(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def fetch_lm_head(out: Path) -> None:
    index = json.loads((out / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][LM_HEAD]
    header_len = struct.unpack("<Q", fetch(shard, 0, 7))[0]
    header = json.loads(fetch(shard, 8, 8 + header_len - 1))
    meta = header[LM_HEAD]
    if meta["dtype"] != "BF16":
        raise SystemExit(f"{LM_HEAD} is {meta['dtype']}, expected BF16")
    begin, finish = meta["data_offsets"]
    size = finish - begin
    entries = {LM_HEAD: {"dtype": "BF16", "shape": meta["shape"], "data_offsets": [0, size]},
               "__metadata__": {"format": "pt"}}
    head = json.dumps(entries, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    target = out / "lm_head.safetensors"
    if target.exists() and target.stat().st_size == 8 + len(head) + size:
        print(f"{target.name}: already present")
        return
    part = target.with_name(target.name + ".part")
    started = time.time()
    with open(part, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        base = 8 + header_len + begin
        for offset in range(0, size, CHUNK):
            f.write(fetch(shard, base + offset, base + min(offset + CHUNK, size) - 1))
            done = min(offset + CHUNK, size)
            print(f"\r{target.name}: {done / 2**20:,.0f} of {size / 2**20:,.0f} MiB "
                  f"({done / 2**20 / max(time.time() - started, 1e-3):.0f} MiB/s)", end="", flush=True)
    print()
    os.replace(part, target)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES + [HEAD_FILE]:
        target = out / name
        if target.exists() and target.stat().st_size > 0:
            print(f"{name}: already present")
            continue
        data = fetch(name)
        save(target, data)
        print(f"{name}: {len(data):,} bytes")
    fetch_lm_head(out)
    print(f"done: {out} (Clef at revision {CLEF_REVISION[:8]})")


if __name__ == "__main__":
    main()
