"""Read rows of the model's output-embedding matrix (lm_head.weight) without loading all of it.

The scoring head averages the output-embedding rows of each option's tokens. The matrix is
248,320 x 5,120 bf16 (2.5 GB), and a request touches a few hundred rows, so it is memory-mapped
and only those rows are read. It is taken from lm_head.safetensors when the release directory has
one (fetch_release.py writes it), and otherwise from the shard of the full release that holds it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

NAME = "lm_head.weight"


class OutputEmbeddingRows:
    def __init__(self, release: str | Path) -> None:
        release = Path(release)
        shard = release / "lm_head.safetensors"
        if not shard.exists():
            index = json.loads((release / "model.safetensors.index.json").read_text())
            shard = release / index["weight_map"][NAME]
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        meta = header[NAME]
        if meta["dtype"] != "BF16":
            raise ValueError(f"{NAME} is {meta['dtype']}, expected BF16")
        start, end = meta["data_offsets"]
        rows, cols = meta["shape"]
        if end - start != rows * cols * 2:
            raise ValueError(f"{NAME} size does not match its shape")
        self.shape = (rows, cols)
        self.matrix = np.memmap(shard, dtype=np.uint16, mode="r", offset=8 + header_len + start, shape=(rows, cols))

    def rows(self, token_ids: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        bits = np.ascontiguousarray(self.matrix[np.asarray(token_ids, dtype=np.int64)])
        tensor = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
        return tensor.to(dtype)

    def as_tensor_lookup(self, dtype: torch.dtype) -> "_Lookup":
        """An object the head can index like a weight tensor: lookup[token_id_tensor] -> rows."""
        return _Lookup(self, dtype)


class _Lookup:
    def __init__(self, source: OutputEmbeddingRows, dtype: torch.dtype) -> None:
        self.source, self.dtype = source, dtype

    def __getitem__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.source.rows(token_ids.detach().cpu().numpy(), self.dtype)
