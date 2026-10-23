"""Sharpness-Aware Minimization (Foret et al., 2020), Algorithm 1.

min_w  max_{||eps||_p <= rho}  L_S(w + eps)
eps_hat(w) = rho * sign(g) |g|^{q-1} / (||g||_q^q)^{1/p},  with p = 2:  eps_hat = rho * g / ||g||_2.
The SAM gradient is approximated by grad L_S(w) evaluated at w + eps_hat(w) (second-order terms dropped).
"""
from __future__ import annotations

from typing import Callable

import torch


class SAM(torch.optim.Optimizer):
    """Wraps a base optimizer. Usage per step:

        loss_fn().backward(); opt.first_step(zero_grad=True)
        loss_fn().backward(); opt.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        norms = [p.grad.norm(2) for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), 2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        norm = self._grad_norm()
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        if closure is None:
            raise ValueError("SAM.step requires a closure that recomputes the loss with backward()")
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss


def sam_step_m_sharpness(model, loss_fn, x, y, opt: SAM, m: int) -> float:
    """m-sharpness (Section 4.1): each disjoint chunk of m examples gets its own eps_hat (as with one
    accelerator per chunk); the resulting SAM gradients are averaged before the base update."""
    params = [p for g in opt.param_groups for p in g["params"]]
    acc = [torch.zeros_like(p) for p in params]
    chunks = list(zip(x.split(m), y.split(m)))
    total = 0.0
    for xc, yc in chunks:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xc), yc)
        total += loss.item()
        loss.backward()
        opt.first_step(zero_grad=True)
        loss_fn(model(xc), yc).backward()
        with torch.no_grad():
            for a, p in zip(acc, params):
                if p.grad is not None:
                    a.add_(p.grad, alpha=1.0 / len(chunks))
                if "e_w" in opt.state[p]:
                    p.sub_(opt.state[p].pop("e_w"))
    for a, p in zip(acc, params):
        p.grad = a
    opt.base_optimizer.step()
    opt.zero_grad(set_to_none=True)
    return total / len(chunks)
