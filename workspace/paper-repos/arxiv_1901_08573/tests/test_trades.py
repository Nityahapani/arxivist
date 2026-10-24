import torch

from trades_repro.model import WideResNet
from trades_repro.trades import kl_div_batchmean, pgd_attack, trades_loss, trades_perturb
import torch.nn.functional as F


def small():
    torch.manual_seed(0)
    return WideResNet(10, 1, 3)


def test_wrn34_10_parameter_count():
    n = sum(p.numel() for p in WideResNet(34, 10, 10).parameters())
    assert 46.1e6 < n < 46.3e6  # WRN-34-10 ~46.2M


def test_kl_zero_for_identical_logits():
    z = torch.randn(4, 3)
    assert kl_div_batchmean(z, z).abs() < 1e-6


def test_perturbation_respects_eps_and_range():
    m = small()
    x = torch.rand(4, 3, 16, 16)
    xa = trades_perturb(m, x, eps=0.031, steps=3)
    assert (xa - x).abs().max() <= 0.031 + 1e-6 and xa.min() >= 0 and xa.max() <= 1


def test_trades_loss_decomposition():
    m = small().eval()
    x, y = torch.rand(4, 3, 16, 16), torch.randint(0, 3, (4,))
    total, nat, rob = trades_loss(m, x, y, beta=6.0, steps=2)
    assert rob >= -1e-6
    assert torch.isclose(total, nat + 6.0 * rob, atol=1e-5)


def test_pgd_increases_loss():
    m = small().eval()
    x, y = torch.rand(8, 3, 16, 16), torch.randint(0, 3, (8,))
    xa = pgd_attack(m, x, y, steps=5, random_start=False)
    with torch.no_grad():
        assert F.cross_entropy(m(xa), y) >= F.cross_entropy(m(x), y)
    assert (xa - x).abs().max() <= 0.031 + 1e-6
