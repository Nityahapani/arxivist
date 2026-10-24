"""Multi-scale retention (Sun et al., 2023, Section 2).

Three equivalent computation paradigms for one head with decay gamma:
  parallel   (Eq. 5):  Retention(X) = (Q K^T ⊙ D) V,      D_nm = gamma^(n-m) if n >= m else 0
  recurrent  (Eq. 6):  S_n = gamma S_{n-1} + K_n^T V_n,   Retention(X_n) = Q_n S_n
  chunkwise  (Eq. 7):  R_i = K_[i]^T (V_[i] ⊙ zeta) + gamma^B R_{i-1}
                       out_[i] = (Q_[i] K_[i]^T ⊙ D) V_[i] + (Q_[i] R_{i-1}) ⊙ xi
Q and K carry the xPos-style rotation Theta_n = e^{i n theta} (Eq. 3-4), implemented as RoPE.
Tensors are [B, H, T, d]; gamma is [H].
"""
from __future__ import annotations

import torch


def head_gammas(n_heads: int) -> torch.Tensor:
    """gamma = 1 - 2^(-5 - arange(0, h)) (Section 2.2)."""
    return 1 - 2.0 ** (-5 - torch.arange(n_heads, dtype=torch.float64))


def rotate(x: torch.Tensor, offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    T, d = x.shape[-2:]
    th = base ** (-torch.arange(0, d, 2, dtype=torch.float64, device=x.device) / d)
    ang = (torch.arange(T, device=x.device, dtype=torch.float64) + offset)[:, None] * th[None]
    cos, sin = ang.cos().repeat_interleave(2, -1).to(x.dtype), ang.sin().repeat_interleave(2, -1).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return x * cos + torch.stack([-x2, x1], -1).flatten(-2) * sin


def decay_mask(T: int, gamma: torch.Tensor, device=None, dtype=torch.float32) -> torch.Tensor:
    n = torch.arange(T, device=device)
    diff = (n[:, None] - n[None, :]).to(torch.float64)
    D = gamma.to(device)[:, None, None] ** diff.clamp_min(0)
    return (D * (diff >= 0)).to(dtype)  # [H, T, T]


def retention_parallel(q, k, v, gamma):
    D = decay_mask(q.size(-2), gamma, q.device, q.dtype)
    return ((q @ k.transpose(-1, -2)) * D) @ v


def retention_recurrent_step(q_n, k_n, v_n, state, gamma):
    """q_n, k_n: [B, H, d_k]; v_n: [B, H, d_v]; state: [B, H, d_k, d_v]."""
    state = gamma.to(q_n.dtype).view(1, -1, 1, 1) * state + k_n.unsqueeze(-1) * v_n.unsqueeze(-2)
    return torch.einsum("bhk,bhkv->bhv", q_n, state), state


def retention_recurrent(q, k, v, gamma, state=None):
    B, H, T, dk = q.shape
    state = torch.zeros(B, H, dk, v.size(-1), dtype=q.dtype, device=q.device) if state is None else state
    outs = []
    for n in range(T):
        o, state = retention_recurrent_step(q[:, :, n], k[:, :, n], v[:, :, n], state, gamma)
        outs.append(o)
    return torch.stack(outs, 2), state


def retention_chunkwise(q, k, v, gamma, chunk: int):
    B, H, T, dk = q.shape
    g = gamma.to(q.dtype).view(1, H, 1, 1)
    R = torch.zeros(B, H, dk, v.size(-1), dtype=q.dtype, device=q.device)
    outs = []
    for s in range(0, T, chunk):
        qc, kc, vc = q[:, :, s:s + chunk], k[:, :, s:s + chunk], v[:, :, s:s + chunk]
        Bc = qc.size(2)
        j = torch.arange(Bc, device=q.device, dtype=q.dtype).view(1, 1, -1, 1)
        inner = ((qc @ kc.transpose(-1, -2)) * decay_mask(Bc, gamma, q.device, q.dtype)) @ vc
        cross = (qc @ R) * g ** (j + 1)                       # xi_j = gamma^(j+1)
        outs.append(inner + cross)
        zeta = g ** (Bc - 1 - j)                              # zeta_j = gamma^(B-1-j)
        R = kc.transpose(-1, -2) @ (vc * zeta) + g ** Bc * R
    return torch.cat(outs, 2), R
