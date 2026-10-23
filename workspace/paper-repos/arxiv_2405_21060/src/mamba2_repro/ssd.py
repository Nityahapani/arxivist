"""Structured State Space Duality (Dao & Gu, 2024).

Scalar-identity SSM per head (Section 3-4):   h_t = a_t h_{t-1} + B_t x_t^T,   y_t = h_t^T C_t
with h_t in R^{N x P}. Equivalent forms:
  * recurrent (linear scan),
  * quadratic / "attention" form (Section 4, Eq. 16-17): Y = (L ∘ C B^T) X, L_ij = a_{j+1} ... a_i (i >= j),
  * chunked SSD algorithm (Section 6, Listing 1): diagonal blocks quadratic, off-diagonal blocks via
    low-rank chunk states and a short recurrence over chunks.
Shapes: X [b, l, h, p], log_a ("A" in Listing 1, = dt * A <= 0) [b, l, h], B and C [b, l, h, n].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: out[..., i, j] = sum_{k=j+1}^{i} x[..., k] for i >= j, else -inf."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    x = x.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1), 0)
    s = torch.cumsum(x, dim=-2)
    return s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), 0), float("-inf"))


def ssd_recurrent(X, log_a, B, C, initial_state=None):
    b, l, h, p = X.shape
    n = B.size(-1)
    state = X.new_zeros(b, h, p, n) if initial_state is None else initial_state
    ys = []
    for t in range(l):
        state = torch.exp(log_a[:, t])[..., None, None] * state + X[:, t, :, :, None] * B[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
    return torch.stack(ys, 1), state


def ssd_quadratic(X, log_a, B, C):
    L = torch.exp(segsum(log_a.transpose(1, 2)))                  # [b, h, l, l]
    scores = torch.einsum("blhn,bshn->bhls", C, B) * L
    return torch.einsum("bhls,bshp->blhp", scores, X)


def ssd_chunked(X, log_a, B, C, chunk: int, initial_state=None):
    b, l, h, p = X.shape
    assert l % chunk == 0, "sequence length must be a multiple of the chunk size"
    c = l // chunk
    X, B, C = (t.reshape(b, c, chunk, h, -1) for t in (X, B, C))
    A = log_a.reshape(b, c, chunk, h).permute(0, 3, 1, 2)            # [b, h, c, chunk]
    A_cum = torch.cumsum(A, -1)
    # 1. intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    # 2. state at the end of each chunk
    decay_states = torch.exp(A_cum[..., -1:] - A_cum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    # 3. inter-chunk recurrence
    if initial_state is None:
        initial_state = torch.zeros_like(states[:, :1])
    else:
        initial_state = initial_state.unsqueeze(1)
    states = torch.cat([initial_state, states], 1)
    decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))   # [b, h, c+1, c+1]
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    # 4. state -> output (off-diagonal blocks)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(A_cum))
    return (Y_diag + Y_off).reshape(b, l, h, p), final_state
