"""HBM access accounting (Section 3.2, Theorem 2).

Standard attention (Algorithm 0) requires Theta(N d + N^2) HBM accesses; FlashAttention requires
Theta(N^2 d^2 / M) for SRAM size M (d <= M <= N d). Counts are in elements, per (batch, head).
"""
from __future__ import annotations

import math

from .flash import block_sizes


def standard_hbm_accesses(N: int, d: int) -> int:
    """Algorithm 0: read Q,K (2Nd), write S (N^2); read S, write P (2N^2); read P,V (N^2 + Nd), write O (Nd)."""
    return 4 * N * d + 4 * N * N


def flash_hbm_accesses(N: int, d: int, M: int) -> int:
    """Algorithm 1 (loop order of Algorithm 2): K_j, V_j loaded once per row block; Q_i, O_i, l_i, m_i per row block."""
    br, bc = block_sizes(M, d)
    Tr = math.ceil(N / br)
    return Tr * 2 * N * d + N * d + N * d + 2 * N   # K,V per row block; read Q; write O; write L


def flash_theorem2_order(N: int, d: int, M: int) -> float:
    return N * N * d * d / M
