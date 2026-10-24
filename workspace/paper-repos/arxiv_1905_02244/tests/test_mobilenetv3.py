import torch

from mobilenetv3_repro.model import LARGE, SMALL, Bneck, MobileNetV3, h_swish, make_divisible


def n(m):
    return sum(p.numel() for p in m.parameters())


def test_h_swish_definition():
    x = torch.linspace(-5, 5, 101)
    assert torch.allclose(h_swish(x), x * torch.clamp(x + 3, 0, 6) / 6)
    assert torch.all(h_swish(torch.tensor([-3.0, -4.0])) == 0)


def test_parameter_counts():
    assert abs(n(MobileNetV3("large")) / 1e6 - 5.4) < 0.1   # Table 3: 5.4M
    assert abs(n(MobileNetV3("small")) / 1e6 - 2.5) < 0.1   # Table 4: 2.5M


def test_table_lengths_and_output():
    assert len(LARGE) == 15 and len(SMALL) == 11
    assert MobileNetV3("small", 10)(torch.rand(2, 3, 64, 64)).shape == (2, 10)


def test_downsampling_factor():
    m = MobileNetV3("large").eval()
    assert m.features(torch.rand(1, 3, 224, 224)).shape == (1, 960, 7, 7)


def test_bneck_residual():
    assert Bneck(16, 3, 16, 16, False, "RE", 1).residual and not Bneck(16, 3, 64, 24, False, "RE", 2).residual
    assert make_divisible(72 // 4) == 24 and make_divisible(240 // 4) == 64
