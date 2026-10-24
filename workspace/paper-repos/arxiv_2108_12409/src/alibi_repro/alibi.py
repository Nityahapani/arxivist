"""Attention with Linear Biases (Press et al., Section 3).

softmax(q_i K^T + m · [-(i-1), ..., -2, -1, 0]) — no position embeddings are added to the input.
Head-specific slopes m form a geometric sequence starting at 2^(-8/n) with ratio 2^(-8/n).
"""
from __future__ import annotations

import math

import torch


def get_slopes(n_heads: int) -> list[float]:
    """Slopes for n heads; for non-powers of two, interleave from the next power of two (paper Sec. 3
    footnote / reference implementation)."""
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return pow2(closest) + get_slopes(2 * closest)[0::2][: n_heads - closest]


def alibi_bias(n_heads: int, q_len: int, k_len: int | None = None, device=None) -> torch.Tensor:
    """Bias tensor [H, q_len, k_len] with entry -m_h * (i - j) for causal positions j <= i.

    Query position i is aligned to the end of the key sequence (supports cached decoding)."""
    k_len = k_len or q_len
    slopes = torch.tensor(get_slopes(n_heads), device=device).view(-1, 1, 1)
    i = torch.arange(k_len - q_len, k_len, device=device).view(-1, 1)
    j = torch.arange(k_len, device=device).view(1, -1)
    dist = (i - j).clamp(min=0).float()
    return -slopes * dist
