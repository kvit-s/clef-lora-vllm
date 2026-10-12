"""Read the hidden-state files written by the clef_capture vLLM plugin.

Layout: b"CLEFHS1\\n", little-endian u64 header length, JSON header, then
(prompt_tokens - first_position) x hidden_size bf16 values for positions first_position onwards.
first_position is above 0 when vLLM served the start of the prompt from its prefix cache (files
without it start at 0). The header's token_ids_sha256 covers all the prompt's ids and is checked
against the ids the service sent, so a file can never be paired with a different prompt.
"""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import torch

MAGIC = b"CLEFHS1\n"


def token_ids_digest(token_ids) -> str:
    return hashlib.sha256(np.asarray(token_ids, dtype="<i4").tobytes()).hexdigest()


def read(path: str, expected_token_ids=None) -> tuple[dict, torch.Tensor]:
    """Return (header, hidden states of positions first_position onwards, as a bf16 tensor of shape
    [prompt_tokens - first_position, hidden_size])."""
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not a clef_capture file")
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        header.setdefault("first_position", 0)
        rows, cols = header["prompt_tokens"] - header["first_position"], header["hidden_size"]
        data = f.read()
    if header.get("dtype") != "bf16" or len(data) != rows * cols * 2:
        raise ValueError(f"{path}: expected {rows}x{cols} bf16 values, found {len(data)} bytes")
    if expected_token_ids is not None and header["token_ids_sha256"] != token_ids_digest(expected_token_ids):
        raise ValueError(f"{path}: hidden states belong to a different prompt")
    bits = np.frombuffer(data, dtype=np.uint16).reshape(rows, cols)
    hidden = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
    return header, hidden
