import torch

from mamba2_repro.model import Mamba2Block, Mamba2LM
from mamba2_repro.ssd import segsum, ssd_chunked, ssd_quadratic, ssd_recurrent


def inputs(b=2, l=16, h=3, p=4, n=5):
    torch.manual_seed(0)
    X = torch.randn(b, l, h, p, dtype=torch.float64)
    log_a = -torch.rand(b, l, h, dtype=torch.float64)
    B, C = torch.randn(b, l, h, n, dtype=torch.float64), torch.randn(b, l, h, n, dtype=torch.float64)
    return X, log_a, B, C


def test_segsum():
    x = torch.tensor([1.0, 2.0, 3.0])
    s = segsum(x)
    assert s[2, 0] == 5 and s[1, 1] == 0 and torch.isinf(s[0, 1])


def test_recurrent_equals_quadratic():
    X, a, B, C = inputs()
    assert torch.allclose(ssd_recurrent(X, a, B, C)[0], ssd_quadratic(X, a, B, C), atol=1e-10)


def test_chunked_equals_recurrent_including_final_state():
    X, a, B, C = inputs()
    yr, sr = ssd_recurrent(X, a, B, C)
    for chunk in (1, 4, 8, 16):
        yc, sc = ssd_chunked(X, a, B, C, chunk)
        assert torch.allclose(yc, yr, atol=1e-10) and torch.allclose(sc, sr, atol=1e-10)


def test_chunked_with_initial_state():
    X, a, B, C = inputs()
    s0 = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    assert torch.allclose(ssd_chunked(X, a, B, C, 4, s0)[0], ssd_recurrent(X, a, B, C, s0)[0], atol=1e-10)


def test_block_modes_and_step_decoding():
    torch.manual_seed(0)
    blk = Mamba2Block(16, d_state=8, head_dim=8, chunk=4).double().eval()
    u = torch.randn(2, 10, 16, dtype=torch.float64)
    full = blk(u, "chunked")
    assert torch.allclose(full, blk(u, "recurrent"), atol=1e-9)
    cache = None
    for t in range(10):
        o, cache = blk.step(u[:, t:t + 1], cache)
        assert torch.allclose(o[:, 0], full[:, t], atol=1e-9)


def test_lm_forward():
    m = Mamba2LM(30, 32, 2, 16, 16, 2, 8)
    assert m(torch.randint(0, 30, (2, 13))).shape == (2, 13, 30)
