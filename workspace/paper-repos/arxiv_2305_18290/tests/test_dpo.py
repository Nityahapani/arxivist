import math

import torch

from dpo_repro.data import response_mask
from dpo_repro.dpo import dpo_gradient_weight, dpo_loss, sequence_logps
from dpo_repro.model import TinyGPT


def test_loss_is_log2_when_policy_equals_reference():
    lp = torch.randn(8)
    loss, rw, rl = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6) and torch.all(rw == 0) and torch.all(rl == 0)


def test_gradient_matches_paper_expression():
    pw, pl = torch.tensor(-3.0, requires_grad=True), torch.tensor(-2.0, requires_grad=True)
    rw, rl, beta = torch.tensor(-3.5), torch.tensor(-1.0), 0.5
    loss, _, _ = dpo_loss(pw, pl, rw, rl, beta)
    loss.backward()
    wgt = dpo_gradient_weight(pw, pl, rw, rl, beta).item()
    assert math.isclose(pw.grad.item(), -beta * wgt, rel_tol=1e-5)
    assert math.isclose(pl.grad.item(), beta * wgt, rel_tol=1e-5)


def test_sequence_logps_masks_prompt():
    torch.manual_seed(0)
    m = TinyGPT(10, 16, 1, 2, 32, 12)
    ids = torch.randint(0, 10, (2, 12))
    mask = response_mask(ids, 5)
    lp = sequence_logps(m(ids), ids, mask)
    manual = torch.log_softmax(m(ids), -1)[:, 4:-1].gather(2, ids[:, 5:].unsqueeze(-1)).squeeze(-1).sum(-1)
    assert torch.allclose(lp, manual, atol=1e-5)


def test_tabular_optimum_recovers_rlhf_policy():
    """Minimising the expected DPO loss under Bradley-Terry preferences yields
    pi* ∝ pi_ref exp(r / beta) (Eq. 4-5) for a single-step (bandit) problem."""
    torch.manual_seed(0)
    K, beta = 5, 0.5
    r = torch.randn(K)
    log_ref = torch.log_softmax(torch.randn(K), 0)
    logits = log_ref.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.05)
    p = torch.sigmoid(r[:, None] - r[None, :])  # p(i > j)
    for _ in range(3000):
        h = torch.log_softmax(logits, 0) - log_ref
        z = beta * (h[:, None] - h[None, :])
        loss = -(p * torch.nn.functional.logsigmoid(z)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    target = torch.log_softmax(log_ref + r / beta, 0)
    assert torch.allclose(torch.log_softmax(logits, 0), target, atol=1e-2)
