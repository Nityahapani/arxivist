"""Rotary Position Embedding (RoPE), RoFormer Section 3.

theta_i = 10000^{-2(i-1)/d}, i = 1..d/2 (Eq. 15). For position m, each 2-D pair (x_{2i-1}, x_{2i}) is
rotated by angle m * theta_i. Three equivalent implementations are provided:
  * `rotation_matrix`: the sparse block-diagonal R^d_{Theta,m} of Eq. 15;
  * `apply_rope_complex`: the complex form f(x, m) = x e^{i m theta} (Eq. 12);
  * `apply_rope`: the computationally efficient element-wise form of Eq. 34.
"""
from __future__ import annotations

import torch


def rope_thetas(d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    return base ** (-torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)


def rope_cos_sin(positions: torch.Tensor, d: int, base: float = 10000.0):
    ang = positions.float()[:, None] * rope_thetas(d, base, positions.device)[None]  # [T, d/2]
    cos = ang.cos().repeat_interleave(2, -1)
    sin = ang.sin().repeat_interleave(2, -1)
    return cos, sin  # [T, d]


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2, x3, x4, ...) -> (-x2, x1, -x4, x3, ...)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([-x2, x1], -1).flatten(-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor | None = None, base: float = 10000.0) -> torch.Tensor:
    """Eq. 34: R x = x ⊙ cos(mθ) + rotate(x) ⊙ sin(mθ). x: [..., T, d]."""
    T, d = x.shape[-2:]
    positions = torch.arange(T, device=x.device) if positions is None else positions
    cos, sin = rope_cos_sin(positions, d, base)
    return x * cos.to(x.dtype) + rotate_pairs(x) * sin.to(x.dtype)


def apply_rope_complex(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    d = x.shape[-1]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], d // 2, 2).contiguous())
    rot = torch.polar(torch.ones(len(positions), d // 2), positions.float()[:, None] * rope_thetas(d, base)[None])
    return torch.view_as_real(xc * rot).flatten(-2).to(x.dtype)


def rotation_matrix(m: int, d: int, base: float = 10000.0) -> torch.Tensor:
    R = torch.zeros(d, d)
    for i, th in enumerate(rope_thetas(d, base)):
        c, s = torch.cos(m * th), torch.sin(m * th)
        R[2 * i, 2 * i], R[2 * i, 2 * i + 1] = c, -s
        R[2 * i + 1, 2 * i], R[2 * i + 1, 2 * i + 1] = s, c
    return R


def long_term_decay_bound(d: int, max_dist: int, base: float = 10000.0) -> torch.Tensor:
    """Relative upper bound (1/(d/2)) * sum_j |S_j|, S_j = sum_{k<j} e^{i r theta_k} (Section 3.4.3, Fig. 2)."""
    th = rope_thetas(d, base)
    r = torch.arange(max_dist + 1, dtype=torch.float32)[:, None]
    S = torch.cumsum(torch.polar(torch.ones_like(r * th), r * th), dim=1)
    return S.abs().mean(1)
