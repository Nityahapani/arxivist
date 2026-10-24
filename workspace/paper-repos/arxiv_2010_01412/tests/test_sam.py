import torch
from torch import nn

from sam_repro.model import WideResNet
from sam_repro.sam import SAM, sam_step_m_sharpness


def _lin():
    torch.manual_seed(0)
    return nn.Linear(5, 3)


def test_first_step_perturbation_has_norm_rho():
    m = _lin()
    opt = SAM(m.parameters(), torch.optim.SGD, rho=0.07, lr=0.1)
    w0 = [p.clone() for p in m.parameters()]
    m(torch.randn(8, 5)).pow(2).sum().backward()
    opt.first_step()
    delta = torch.cat([(p - q).flatten() for p, q in zip(m.parameters(), w0)])
    assert abs(delta.norm().item() - 0.07) < 1e-5


def test_second_step_restores_then_updates_with_perturbed_grad():
    m = _lin()
    x = torch.randn(8, 5)
    w0 = [p.detach().clone() for p in m.parameters()]
    opt = SAM(m.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    m(x).pow(2).sum().backward()
    opt.first_step(zero_grad=True)
    m(x).pow(2).sum().backward()
    g_pert = [p.grad.clone() for p in m.parameters()]
    opt.second_step()
    for p, q, gp in zip(m.parameters(), w0, g_pert):
        assert torch.allclose(p, q - 0.1 * gp, atol=1e-6)


def test_closure_step_reduces_loss():
    m = _lin()
    x, y = torch.randn(32, 5), torch.randn(32, 3)
    opt = SAM(m.parameters(), torch.optim.SGD, rho=0.05, lr=0.05)
    def closure():
        l = ((m(x) - y) ** 2).mean(); l.backward(); return l
    first = closure().item(); opt.zero_grad()
    for _ in range(50):
        opt.zero_grad(); opt.step(closure)
    assert ((m(x) - y) ** 2).mean().item() < first


def test_m_sharpness_full_batch_equals_standard_sam():
    x, y = torch.randn(8, 5), torch.randn(8, 3)
    loss_fn = lambda o, t: ((o - t) ** 2).mean()
    a, b = _lin(), _lin()
    oa = SAM(a.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    ob = SAM(b.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    sam_step_m_sharpness(a, loss_fn, x, y, oa, m=8)
    def closure():
        l = loss_fn(b(x), y); l.backward(); return l
    ob.step(closure)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_wrn28_10_size():
    n = sum(p.numel() for p in WideResNet(28, 10, 10).parameters())
    assert 36.4e6 < n < 36.6e6  # WRN-28-10 ~36.5M
