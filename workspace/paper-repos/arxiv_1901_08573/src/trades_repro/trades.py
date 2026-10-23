"""TRADES objective (Section 4, Eq. 5) and attacks used for training and evaluation.

min_f E[ CE(f(X), Y) + beta * max_{X' in B(X, eps)} KL(f(X) || f(X')) ],   beta = 1/lambda.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_div_batchmean(logits_clean: torch.Tensor, logits_adv: torch.Tensor) -> torch.Tensor:
    """KL(p_clean || p_adv) averaged over the batch."""
    return F.kl_div(F.log_softmax(logits_adv, 1), F.softmax(logits_clean, 1), reduction="batchmean")


def trades_perturb(model, x, eps=0.031, step_size=0.007, steps=10, init_std=0.001, generator=None):
    """Inner maximisation of KL(f(x) || f(x')) by projected sign-gradient ascent from x + 0.001·N(0, I)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p_clean = model(x)
    noise = torch.randn(x.shape, generator=generator).to(x.device) if generator is not None else torch.randn_like(x)
    x_adv = (x + init_std * noise).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = kl_div_batchmean(p_clean, model(x_adv))
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    model.train(was_training)
    return x_adv.detach()


def trades_loss(model, x, y, beta=6.0, **attack):
    x_adv = trades_perturb(model, x, **attack)
    logits = model(x)
    natural = F.cross_entropy(logits, y)
    robust = kl_div_batchmean(logits, model(x_adv))
    return natural + beta * robust, natural.detach(), robust.detach()


def pgd_attack(model, x, y, eps=0.031, step_size=0.003, steps=20, random_start=True, generator=None):
    """White-box L_inf PGD on cross-entropy (FGSM^k evaluation, Section 5)."""
    model.eval()
    x_adv = x.clone()
    if random_start:
        u = torch.rand(x.shape, generator=generator).to(x.device) if generator is not None else torch.rand_like(x)
        x_adv = (x + (2 * u - 1) * eps).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(x_adv), y), x_adv)
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()
