import torch

from retnet_repro.model import RetNet
from retnet_repro.retention import head_gammas, retention_chunkwise, retention_parallel, retention_recurrent


def qkv(B=2, H=3, T=13, dk=4, dv=6):
    torch.manual_seed(0)
    return torch.randn(B, H, T, dk, dtype=torch.float64), torch.randn(B, H, T, dk, dtype=torch.float64), torch.randn(B, H, T, dv, dtype=torch.float64)


def test_gammas():
    g = head_gammas(4)
    assert torch.allclose(g, torch.tensor([1 - 1 / 32, 1 - 1 / 64, 1 - 1 / 128, 1 - 1 / 256], dtype=torch.float64))


def test_parallel_equals_recurrent():
    q, k, v = qkv()
    g = head_gammas(3)
    assert torch.allclose(retention_parallel(q, k, v, g), retention_recurrent(q, k, v, g)[0], atol=1e-10)


def test_chunkwise_equals_parallel_and_state():
    q, k, v = qkv()
    g = head_gammas(3)
    par = retention_parallel(q, k, v, g)
    for chunk in (1, 4, 5, 13):
        out, R = retention_chunkwise(q, k, v, g, chunk)
        assert torch.allclose(out, par, atol=1e-10)
        assert torch.allclose(R, retention_recurrent(q, k, v, g)[1], atol=1e-10)


def test_model_modes_agree_and_recurrent_decoding_matches():
    torch.manual_seed(0)
    m = RetNet(20, 16, 2, 2, 32).double().eval()
    ids = torch.randint(0, 20, (2, 11))
    par = m(ids)
    assert torch.allclose(par, m(ids, "chunkwise", 4), atol=1e-9)
    states = [None] * 2
    for t in range(11):
        logits, states = m.step(ids[:, t:t + 1], states, t)
        assert torch.allclose(logits[:, 0], par[:, t], atol=1e-9)


def test_state_size_independent_of_length():
    m = RetNet(20, 16, 2, 2, 32).eval()
    states = [None] * 2
    for t in range(50):
        _, states = m.step(torch.zeros(1, 1, dtype=torch.long), states, t)
    assert states[0].shape == (1, 2, 8, 16)
