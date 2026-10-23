import pytest
import torch

from flashattention_repro.flash import block_sizes, flash_attention, flash_forward, standard_attention
from flashattention_repro.io import flash_hbm_accesses, standard_hbm_accesses


def qkv(N=37, d=8, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(2, 3, N, d, dtype=torch.float64, requires_grad=True) for _ in range(3)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("br,bc", [(8, 8), (5, 11), (64, 64)])
def test_forward_exact(causal, br, bc):
    q, k, v = qkv()
    o, L = flash_forward(q, k, v, br, bc, causal)
    assert torch.allclose(o, standard_attention(q, k, v, causal), atol=1e-10)
    S = q @ k.transpose(-1, -2) * 8 ** -0.5
    if causal:
        S = S.masked_fill(torch.ones(37, 37, dtype=torch.bool).triu(1), float("-inf"))
    assert torch.allclose(L, torch.logsumexp(S, -1), atol=1e-10)


@pytest.mark.parametrize("causal", [False, True])
def test_backward_exact(causal):
    q, k, v = qkv()
    do = torch.randn(2, 3, 37, 8, dtype=torch.float64)
    g1 = torch.autograd.grad(flash_attention(q, k, v, 8, 6, causal), (q, k, v), do)
    g2 = torch.autograd.grad(standard_attention(q, k, v, causal), (q, k, v), do)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-9)


def test_block_sparse_matches_masked_dense():
    q, k, v = qkv(N=32)
    bm = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
    dense_mask = bm.repeat_interleave(8, 0).repeat_interleave(8, 1)
    o, _ = flash_forward(q, k, v, 8, 8, block_mask=bm)
    assert torch.allclose(o, standard_attention(q, k, v, mask=dense_mask), atol=1e-10)


def test_block_sizes_and_io_scaling():
    assert block_sizes(M=24576, d=64) == (64, 96)
    N, d, M = 4096, 64, 24576
    assert flash_hbm_accesses(N, d, M) < standard_hbm_accesses(N, d)
    # Theorem 2: while B_r < d, quadrupling M cuts FlashAttention's dominant HBM term ~4x
    assert flash_hbm_accesses(N, d, 16384) < 0.3 * flash_hbm_accesses(N, d, 4096)
