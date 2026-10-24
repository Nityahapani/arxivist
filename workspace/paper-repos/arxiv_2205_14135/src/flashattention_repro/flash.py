"""FlashAttention reference implementation in PyTorch (Dao et al., 2022).

Algorithm 1 / 2 (forward): tile Q into row blocks of size B_r and K, V into column blocks of size B_c;
keep running row max m and normaliser l (online softmax) so the N x N matrix is never materialised.
Algorithm 4 (backward): recompute S and P block by block from Q, K, V and the saved log-sum-exp
L = m + log(l), using D = rowsum(dO ∘ O).
Algorithm 5: block-sparse variant that skips zero blocks of a block mask.

This is an *algorithmic* reference (Python loops over tiles), meant for correctness and IO accounting,
not speed. Tensors are [B, H, N, d].
"""
from __future__ import annotations

import math

import torch


def block_sizes(M: int, d: int) -> tuple[int, int]:
    """B_c = ceil(M / 4d), B_r = min(ceil(M / 4d), d) for on-chip SRAM of size M (Algorithm 1, line 1)."""
    bc = math.ceil(M / (4 * d))
    return min(bc, d), bc


def _mask_block(S, i0, j0, causal, block_mask, br, bc):
    if causal:
        r = torch.arange(i0, i0 + S.size(-2), device=S.device)[:, None]
        c = torch.arange(j0, j0 + S.size(-1), device=S.device)[None]
        S = S.masked_fill(c > r, float("-inf"))
    return S


def flash_forward(q, k, v, br=64, bc=64, causal=False, scale=None, block_mask=None, counter=None):
    """Returns O [B, H, N, d] and L = logsumexp(S) [B, H, N]."""
    B, H, N, d = q.shape
    scale = scale if scale is not None else d ** -0.5
    O = torch.zeros_like(q)
    L = torch.empty(B, H, N, dtype=q.dtype, device=q.device)
    for i0 in range(0, N, br):
        qi = q[:, :, i0:i0 + br] * scale
        m = torch.full(qi.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros_like(qi)
        if counter is not None:
            counter["read"] += qi[0, 0].numel() * 1  # Q_i
        for j0 in range(0, N, bc):
            if causal and j0 > i0 + qi.size(-2) - 1:
                break
            if block_mask is not None and not block_mask[i0 // br, j0 // bc]:
                continue
            kj, vj = k[:, :, j0:j0 + bc], v[:, :, j0:j0 + bc]
            if counter is not None:
                counter["read"] += kj[0, 0].numel() + vj[0, 0].numel()
            S = _mask_block(qi @ kj.transpose(-1, -2), i0, j0, causal, block_mask, br, bc)
            m_new = torch.maximum(m, S.amax(-1))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            P = torch.exp(S - m_safe[..., None])
            alpha = torch.exp(m - m_safe)
            l = alpha * l + P.sum(-1)
            acc = alpha[..., None] * acc + P @ vj
            m = m_new
        O[:, :, i0:i0 + br] = acc / l.clamp_min(1e-30)[..., None]
        L[:, :, i0:i0 + br] = m + torch.log(l.clamp_min(1e-30))
        if counter is not None:
            counter["write"] += qi[0, 0].numel() + qi.size(-2)
    return O, L


def flash_backward(q, k, v, o, L, do, br=64, bc=64, causal=False, scale=None, block_mask=None):
    """Algorithm 4: recompute attention probabilities tile by tile. Returns dQ, dK, dV."""
    d = q.size(-1)
    scale = scale if scale is not None else d ** -0.5
    N = q.size(-2)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Dvec = (do * o).sum(-1)                                   # D_i = rowsum(dO_i ∘ O_i)
    for j0 in range(0, N, bc):
        kj, vj = k[:, :, j0:j0 + bc], v[:, :, j0:j0 + bc]
        dkj, dvj = torch.zeros_like(kj), torch.zeros_like(vj)
        for i0 in range(0, N, br):
            if causal and j0 > i0 + br - 1:
                continue
            if block_mask is not None and not block_mask[i0 // br, j0 // bc]:
                continue
            qi, doi = q[:, :, i0:i0 + br], do[:, :, i0:i0 + br]
            S = _mask_block((qi * scale) @ kj.transpose(-1, -2), i0, j0, causal, block_mask, br, bc)
            P = torch.exp(S - L[:, :, i0:i0 + br, None])
            dvj += P.transpose(-1, -2) @ doi
            dP = doi @ vj.transpose(-1, -2)
            dS = P * (dP - Dvec[:, :, i0:i0 + br, None])
            dq[:, :, i0:i0 + br] += scale * dS @ kj
            dkj += scale * dS.transpose(-1, -2) @ qi
        dk[:, :, j0:j0 + bc], dv[:, :, j0:j0 + bc] = dkj, dvj
    return dq, dk, dv


class FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, br, bc, causal, block_mask):
        o, L = flash_forward(q, k, v, br, bc, causal, block_mask=block_mask)
        ctx.save_for_backward(q, k, v, o, L)  # O(N) extra memory: only L, not the N x N matrix
        ctx.cfg = (br, bc, causal, block_mask)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        br, bc, causal, block_mask = ctx.cfg
        dq, dk, dv = flash_backward(q, k, v, o, L, do, br, bc, causal, block_mask=block_mask)
        return dq, dk, dv, None, None, None, None


def flash_attention(q, k, v, br=64, bc=64, causal=False, block_mask=None):
    return FlashAttentionFn.apply(q, k, v, br, bc, causal, block_mask)


def standard_attention(q, k, v, causal=False, mask=None):
    S = q @ k.transpose(-1, -2) * q.size(-1) ** -0.5
    if causal:
        N = q.size(-2)
        S = S.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    if mask is not None:
        S = S.masked_fill(~mask, float("-inf"))
    return torch.softmax(S, -1) @ v
