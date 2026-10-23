import math

import torch

from byol_repro.augment import BYOLAugment, solarize
from byol_repro.loss import byol_loss, regression_loss
from byol_repro.model import BYOL
from byol_repro.schedule import tau_schedule


def small():
    return BYOL("resnet18", 1, True, 8, hidden=32, proj_dim=8)


def test_regression_loss_is_2_minus_2cos():
    p, z = torch.randn(5, 8), torch.randn(5, 8)
    cos = torch.nn.functional.cosine_similarity(p, z)
    assert torch.allclose(regression_loss(p, z), 2 - 2 * cos, atol=1e-6)
    assert regression_loss(z, 3 * z).abs().max() < 1e-6


def test_tau_schedule_endpoints():
    assert math.isclose(tau_schedule(0, 100), 0.996) and math.isclose(tau_schedule(100, 100), 1.0)


def test_target_init_copy_and_no_grad():
    m = small()
    assert all(not p.requires_grad for p in m.target_parameters())
    for a, b in zip(m.encoder.parameters(), m.target_encoder.parameters()):
        assert torch.equal(a, b)


def test_gradients_flow_only_through_online():
    m = small()
    x1, x2 = torch.rand(4, 3, 16, 16), torch.rand(4, 3, 16, 16)
    byol_loss(m.online(x1), m.online(x2), m.target(x1), m.target(x2)).backward()
    assert m.predictor[0].weight.grad is not None
    assert all(p.grad is None for p in m.target_parameters())


def test_ema_update():
    m = small()
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in m.target_encoder.parameters()]
    m.update_target(0.9)
    for pt, b, po in zip(m.target_encoder.parameters(), before, m.encoder.parameters()):
        assert torch.allclose(pt, 0.9 * b + 0.1 * po)


def test_asymmetric_augmentations():
    assert BYOLAugment(32, "T").p == {"blur": 1.0, "solarize": 0.0}
    assert BYOLAugment(32, "T_prime").p == {"blur": 0.1, "solarize": 0.2}
    x = torch.full((2, 3, 4, 4), 0.8)
    assert torch.allclose(solarize(x, p=1.0), torch.full_like(x, 0.2))


def test_resnet50_projector_shape():
    m = BYOL("resnet50")
    assert m.projector[0].in_features == 2048 and m.projector[-1].out_features == 256 and m.predictor[0].out_features == 4096
