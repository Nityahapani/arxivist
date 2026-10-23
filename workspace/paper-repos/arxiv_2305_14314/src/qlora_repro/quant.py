"""4-bit NormalFloat (NF4), blockwise quantization and Double Quantization (Dettmers et al., 2023, Section 3).

NF4 data type: quantiles of N(0, 1) normalised to [-1, 1], built asymmetrically so that 0 is exactly
representable (2^{k-1} positive levels and 2^{k-1}-1 negative levels plus zero; Section 3, "4-bit
NormalFloat Quantization"). Weights are quantized in blocks of 64 with an absmax constant per block.
Double Quantization quantizes those FP32 constants again to 8 bits with block size 256 after
subtracting their mean, reducing the overhead from 32/64 = 0.5 to 8/64 + 32/(64*256) ≈ 0.127 bits/param.
"""
from __future__ import annotations

import torch
from scipy.stats import norm

NF4_OFFSET = 0.9677083  # 0.5 * (1 - 1/(2*15) + 1 - 1/(2*16)) : symmetric-quantile end point


def nf4_levels(offset: float = NF4_OFFSET) -> torch.Tensor:
    pos = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1].numpy()).tolist()        # 8 positive
    neg = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1].numpy())).tolist()     # 7 negative
    v = torch.tensor(sorted(pos + [0.0] + neg), dtype=torch.float32)
    return v / v.abs().max()


def fp4_levels() -> torch.Tensor:
    """FP4 (E2M1) magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign, normalised to [-1, 1] (baseline)."""
    mags = torch.tensor([0.0, 0.5, 1, 1.5, 2, 3, 4, 6]) / 6
    return torch.unique(torch.cat([-mags, mags]))


def int4_levels() -> torch.Tensor:
    return torch.arange(-7, 8, dtype=torch.float32) / 7  # symmetric absmax Int4 (15 levels)


def quantize_blockwise(w: torch.Tensor, levels: torch.Tensor, block: int = 64):
    """Returns (codes uint8 [n], absmax fp32 [n_blocks], shape). Pads to a multiple of block."""
    flat = w.detach().float().flatten()
    pad = (-flat.numel()) % block
    flat = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block)
    absmax = flat.abs().amax(1).clamp_min(1e-12)
    normed = flat / absmax[:, None]
    codes = (normed.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(-1).to(torch.uint8)
    return codes.flatten(), absmax, tuple(w.shape)


def dequantize_blockwise(codes, absmax, shape, levels, block: int = 64):
    vals = levels[codes.long()].view(-1, block) * absmax[:, None]
    n = 1
    for s in shape:
        n *= s
    return vals.flatten()[:n].view(shape)


def double_quantize(absmax: torch.Tensor, block: int = 256):
    """8-bit blockwise quantization of the (mean-centred) FP32 quantization constants."""
    mean = absmax.mean()
    c = absmax - mean
    pad = (-c.numel()) % block
    c = torch.cat([c, c.new_zeros(pad)]).view(-1, block)
    scale = c.abs().amax(1).clamp_min(1e-12) / 127.0
    q = torch.round(c / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.flatten(), scale, mean, absmax.numel()


def double_dequantize(q, scale, mean, n, block: int = 256):
    return (q.view(-1, block).float() * scale[:, None]).flatten()[:n] + mean


def bits_per_parameter(block: int = 64, double_quant: bool = True, dq_block: int = 256) -> float:
    if not double_quant:
        return 4 + 32 / block
    return 4 + 8 / block + 32 / (block * dq_block)
