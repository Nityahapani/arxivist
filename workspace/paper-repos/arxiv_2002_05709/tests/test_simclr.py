import math

import torch
import torch.nn.functional as F

from simclr_repro.augment import SimCLRAugment, hsv_to_rgb, rgb_to_hsv
from simclr_repro.lars import LARS, lars_param_groups, warmup_cosine
from simclr_repro.loss import nt_xent
from simclr_repro.model import SimCLR


def manual_nt_xent(z1, z2, tau):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.size(0)
    total = 0.0
    for i in range(2 * n):
        j = (i + n) % (2 * n)
        num = math.exp((z[i] @ z[j]).item() / tau)
        den = sum(math.exp((z[i] @ z[k]).item() / tau) for k in range(2 * n) if k != i)
        total += -math.log(num / den)
    return total / (2 * n)


def test_nt_xent_matches_manual():
    torch.manual_seed(0)
    z1, z2 = torch.randn(4, 8), torch.randn(4, 8)
    assert abs(nt_xent(z1, z2, 0.5).item() - manual_nt_xent(z1, z2, 0.5)) < 1e-5


def test_nt_xent_lower_for_aligned_pairs():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    assert nt_xent(z, z + 0.01 * torch.randn_like(z), 0.1) < nt_xent(z, torch.randn(8, 16), 0.1)


def test_hsv_roundtrip():
    x = torch.rand(2, 3, 5, 5)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-5)


def test_augment_shape_and_range():
    x = torch.rand(4, 3, 32, 32)
    y = SimCLRAugment(24, 1.0, blur=True)(x, torch.Generator().manual_seed(0))
    assert y.shape == (4, 3, 24, 24) and y.min() >= 0 and y.max() <= 1


def test_model_dims():
    m = SimCLR("resnet50", proj_dim=128, cifar_stem=True, base=8)
    h, z = m(torch.rand(2, 3, 32, 32))
    assert h.shape == (2, 8 * 8 * 4) and z.shape == (2, 128)
    full = SimCLR("resnet50", proj_dim=128)
    assert full.encoder.out_dim == 2048
    n = sum(p.numel() for p in full.encoder.parameters())
    assert 23.4e6 < n < 23.6e6  # ResNet-50 without fc: ~23.5M parameters


def test_lars_groups_and_step():
    m = SimCLR("resnet18", proj_dim=8, cifar_stem=True, base=4)
    groups = lars_param_groups(m, 1e-6)
    assert all(p.ndim == 1 for p in groups[1]["params"])
    opt = LARS(groups, lr=0.1)
    _, z = m(torch.rand(4, 3, 16, 16))
    z.pow(2).sum().backward()
    before = [p.clone() for p in m.parameters()]
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert warmup_cosine(0, 100, 10, 1.0) == 0.1 and abs(warmup_cosine(100, 100, 10, 1.0)) < 1e-9
