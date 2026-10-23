import math

import torch

from alibi_repro.alibi import alibi_bias, get_slopes
from alibi_repro.evaluate import nonoverlapping_ppl, sliding_ppl
from alibi_repro.model import TransformerLM


def test_slopes_power_of_two():
    assert get_slopes(8) == [2 ** -(i + 1) for i in range(8)]  # 1/2, 1/4, ..., 1/256
    s16 = get_slopes(16)
    assert math.isclose(s16[0], 2 ** -0.5) and math.isclose(s16[-1], 2 ** -8)


def test_slopes_non_power_of_two():
    s = get_slopes(12)
    assert len(s) == 12
    assert s[:8] == get_slopes(8)
    assert s[8:] == get_slopes(16)[0::2][:4]


def test_bias_values():
    b = alibi_bias(2, 4)
    m = get_slopes(2)
    assert b.shape == (2, 4, 4)
    assert torch.allclose(b[0, 3], torch.tensor([-3 * m[0], -2 * m[0], -m[0], 0.0]))
    assert torch.all(b.diagonal(dim1=1, dim2=2) == 0)


def test_alibi_runs_beyond_training_length():
    torch.manual_seed(0)
    m = TransformerLM(20, d=16, n_layers=1, n_heads=2, ff=32, pos="alibi")
    assert m(torch.randint(0, 20, (1, 300))).shape == (1, 300, 20)


def test_causality():
    torch.manual_seed(0)
    m = TransformerLM(20, d=16, n_layers=2, n_heads=2, ff=32, pos="alibi").eval()
    x = torch.randint(0, 20, (1, 10))
    y = x.clone(); y[0, 7:] = (y[0, 7:] + 1) % 20
    assert torch.allclose(m(x)[0, :7], m(y)[0, :7], atol=1e-5)


def test_eval_functions():
    torch.manual_seed(0)
    m = TransformerLM(20, d=16, n_layers=1, n_heads=2, ff=32, pos="sinusoidal")
    s = torch.randint(0, 20, (200,))
    assert nonoverlapping_ppl(m, s, 16) > 1 and sliding_ppl(m, s, 16, 8) > 1
