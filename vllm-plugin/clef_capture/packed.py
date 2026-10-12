"""Key and value rows stored in NVFP4: 4 bits per number, with scales.

Each row of `width` numbers is stored as width/2 bytes of 4-bit e2m1 codes (two per byte), one FP8
(e4m3) scale per block of 16 numbers, and one fp32 scale for the row, so a position's twelve rows of
1,024 numbers take 12 x (512 + 64 + 4) = 6,960 bytes instead of 49,152 in fp32. The rounding is the one
eval/row_precision.py measured on real prompts of 16,000 to 70,000 tokens: Clef's probabilities moved by
at most 0.0003 and no answer changed. Works on any device.
"""

from __future__ import annotations

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # midpoints between consecutive e2m1 magnitudes
BLOCK = 16


def bytes_per_row(width: int) -> int:
    return width // 2 + width // BLOCK + 4


SLICE = 4096  # positions packed or unpacked at a time, so that working memory stays small


def _pack_rows(x: torch.Tensor, edges: torch.Tensor):
    """x [n, width] float -> (codes [n, width/2] uint8, block scales [n, width/16] float8, row scales [n])."""
    n, width = x.shape
    blocks = x.reshape(n, width // BLOCK, BLOCK)
    row_scale = x.abs().amax(-1).clamp(min=1e-12) / (448 * 6)
    block_scale = (blocks.abs().amax(-1) / 6 / row_scale[:, None]).to(torch.float8_e4m3fn)
    scale = block_scale.float() * row_scale[:, None]
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (blocks / scale[..., None]).clamp(-6, 6)
    magnitude = torch.bucketize(q.abs(), edges, out_int32=True).to(torch.uint8)
    codes = (magnitude | ((q < 0).to(torch.uint8) << 3)).reshape(n, width)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), block_scale, row_scale


def pack(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """rows [..., width] (any float dtype) -> (codes uint8 [..., width/2], block scales float8 [..., width/16],
    row scales float32 [...]), computed SLICE rows at a time."""
    shape, device = rows.shape, rows.device
    flat = rows.reshape(-1, shape[-1])
    edges = torch.tensor(EDGES, device=device)
    codes = torch.empty(flat.shape[0], shape[-1] // 2, dtype=torch.uint8, device=device)
    block_scale = torch.empty(flat.shape[0], shape[-1] // BLOCK, dtype=torch.float8_e4m3fn, device=device)
    row_scale = torch.empty(flat.shape[0], dtype=torch.float32, device=device)
    for start in range(0, flat.shape[0], SLICE):
        end = min(start + SLICE, flat.shape[0])
        codes[start:end], block_scale[start:end], row_scale[start:end] = _pack_rows(flat[start:end].float(), edges)
    return (codes.reshape(*shape[:-1], shape[-1] // 2), block_scale.reshape(*shape[:-1], shape[-1] // BLOCK),
            row_scale.reshape(shape[:-1]))


def unpack(packed: torch.Tensor, block_scale: torch.Tensor, row_scale: torch.Tensor,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The inverse of pack: (codes, block scales, row scales) -> rows [..., width] in dtype, SLICE rows at a time."""
    lookup = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device, dtype=torch.float32)
    half = packed.shape[-1]
    flat, scales, rows = packed.reshape(-1, half), block_scale.reshape(-1, half * 2 // BLOCK), row_scale.reshape(-1)
    out = torch.empty(flat.shape[0], half * 2, dtype=dtype, device=packed.device)
    for start in range(0, flat.shape[0], SLICE):
        end = min(start + SLICE, flat.shape[0])
        part = flat[start:end]
        codes = torch.stack([part & 0x0F, part >> 4], dim=-1).reshape(end - start, half * 2)
        values = lookup[codes.int()].reshape(end - start, -1, BLOCK)
        scale = scales[start:end].float() * rows[start:end, None]
        out[start:end] = (values * scale[..., None]).reshape(end - start, half * 2).to(dtype)
    return out.reshape(*packed.shape[:-1], half * 2)
