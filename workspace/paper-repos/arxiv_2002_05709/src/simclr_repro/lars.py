"""LARS optimizer (You et al., 2017) as used by SimCLR (Section 2.1, Appendix B.1).

Biases and BatchNorm parameters are excluded from weight decay and LARS adaptation.
"""
from __future__ import annotations

import math

import torch


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6, eta=0.001):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, lars=True))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if group["lars"]:
                    d = d.add(p, alpha=group["weight_decay"])
                    pn, dn = torch.norm(p), torch.norm(d)
                    trust = torch.where((pn > 0) & (dn > 0), group["eta"] * pn / dn, torch.ones_like(pn))
                    d = d.mul(trust)
                buf = self.state[p].setdefault("mu", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(d)
                p.add_(buf, alpha=-group["lr"])
        return loss


def lars_param_groups(model, weight_decay):
    adapt, skip = [], []
    for n, p in model.named_parameters():
        (skip if p.ndim == 1 else adapt).append(p)  # biases and BN weights/biases are 1-D
    return [dict(params=adapt, weight_decay=weight_decay, lars=True), dict(params=skip, weight_decay=0.0, lars=False)]


def warmup_cosine(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
