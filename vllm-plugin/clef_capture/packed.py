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


def pack(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """rows [..., width] (float) -> (codes uint8 [..., width/2], block scales float8 [..., width/16],
    row scales float32 [...])."""
    device = rows.device
    x = rows.float()
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    row_scale = x.abs().amax(-1).clamp(min=1e-12) / (448 * 6)
    block_scale = (blocks.abs().amax(-1) / 6 / row_scale[..., None]).to(torch.float8_e4m3fn)
    scale = block_scale.float() * row_scale[..., None]
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (blocks / scale[..., None]).clamp(-6, 6)
    magnitude = torch.bucketize(q.abs(), torch.tensor(EDGES, device=device))
    codes = (magnitude | ((q < 0).to(torch.int64) << 3)).to(torch.uint8).reshape(shape)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed, block_scale, row_scale


def unpack(packed: torch.Tensor, block_scale: torch.Tensor, row_scale: torch.Tensor,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The inverse of pack: (codes, block scales, row scales) -> rows [..., width]."""
    lookup = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device, dtype=torch.float32)
    low, high = packed & 0x0F, packed >> 4
    codes = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    values = lookup[codes.long()]
    scale = block_scale.float() * row_scale[..., None]
    values = values.reshape(*values.shape[:-1], -1, BLOCK) * scale[..., None]
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(dtype)
