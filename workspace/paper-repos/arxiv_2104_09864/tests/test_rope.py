import torch

from roformer_repro.model import RoFormer, linear_attention_rope
from roformer_repro.rope import apply_rope, apply_rope_complex, long_term_decay_bound, rope_thetas, rotation_matrix


def test_three_forms_agree():
    torch.manual_seed(0)
    d, T = 16, 7
    x = torch.randn(T, d)
    pos = torch.arange(T)
    eff = apply_rope(x)
    assert torch.allclose(eff, apply_rope_complex(x, pos), atol=1e-5)
    for m in range(T):
        assert torch.allclose(eff[m], rotation_matrix(m, d) @ x[m], atol=1e-5)


def test_relative_position_property():
    """<f(q, m), f(k, n)> depends only on m - n (Eq. 11/16)."""
    torch.manual_seed(0)
    d = 32
    q, k = torch.randn(d), torch.randn(d)
    def score(m, n):
        Q = apply_rope(q.expand(max(m, n) + 1, d).clone())[m]
        K = apply_rope(k.expand(max(m, n) + 1, d).clone())[n]
        return (Q * K).sum()
    assert torch.allclose(score(5, 2), score(13, 10), atol=1e-4)
    assert torch.allclose(score(3, 9), score(20, 26), atol=1e-4)


def test_norm_preserving_and_thetas():
    x = torch.randn(10, 8)
    assert torch.allclose(apply_rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rope_thetas(8), torch.tensor([1.0, 10000 ** -0.25, 10000 ** -0.5, 10000 ** -0.75]))


def test_long_term_decay():
    b = long_term_decay_bound(128, 256)
    assert b.argmax() == 0 and b[1:20].mean() > b[50:100].mean() > b[200:].mean()


def test_linear_attention_shapes_and_rope_effect():
    q, k, v = torch.randn(3, 2, 2, 9, 8).unbind(0)
    a, b = linear_attention_rope(q, k, v, True), linear_attention_rope(q, k, v, False)
    assert a.shape == (2, 2, 9, 8) and not torch.allclose(a, b)


def test_encoder_forward():
    m = RoFormer(50, 32, 2, 4, 64, "rope")
    assert m(torch.randint(0, 50, (2, 11))).shape == (2, 11, 50)
