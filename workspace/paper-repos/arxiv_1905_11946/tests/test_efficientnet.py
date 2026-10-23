import math

import torch

from efficientnet_repro.model import ALPHA, BETA, GAMMA, MBConv, SCALING, compound_coefficients, efficientnet, round_filters, round_repeats


def n(m):
    return sum(p.numel() for p in m.parameters())


def test_constraint_alpha_beta_gamma():
    assert abs(ALPHA * BETA**2 * GAMMA**2 - 2.0) < 0.1
    d, w, r = compound_coefficients(1)
    assert (d, w, r) == (ALPHA, BETA, GAMMA)


def test_b0_parameter_count():
    assert abs(n(efficientnet("b0")) / 1e6 - 5.3) < 0.05  # Table 2: 5.3M


def test_b1_b3_parameter_counts():
    assert abs(n(efficientnet("b1")) / 1e6 - 7.8) < 0.05   # 7.8M
    assert abs(n(efficientnet("b3")) / 1e6 - 12.0) < 0.3   # 12M


def test_rounding_helpers():
    assert round_filters(32, 1.0) == 32 and round_filters(32, 1.1) == 32 and round_filters(1280, 2.0) == 2560
    assert round_repeats(3, 3.1) == 10 and round_repeats(1, 1.1) == 2


def test_mbconv_residual_and_shape():
    blk = MBConv(16, 16, 6, 3, 1).eval()
    assert blk.residual and blk(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)
    assert MBConv(16, 24, 6, 5, 2)(torch.randn(2, 16, 8, 8)).shape == (2, 24, 4, 4)


def test_resolutions_table():
    assert [SCALING[f"b{i}"][2] for i in range(8)] == [224, 240, 260, 300, 380, 456, 528, 600]
