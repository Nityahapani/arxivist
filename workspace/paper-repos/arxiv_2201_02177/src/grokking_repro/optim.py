"""Optimizer variants used in the paper's ablation study."""

from __future__ import annotations

import math
from typing import Iterable, Literal

import torch
from torch import Tensor

WeightDecayMode = Literal["none", "zero", "init"]


class PaperAdam(torch.optim.Optimizer):
    """Adam with the authors' decoupled decay/noise options.

    ``weight_decay_mode='zero'`` is AdamW decay toward zero. ``'init'`` pulls
    parameters toward their values at the first update. ``update_noise_std`` adds
    Gaussian noise to the Adam update direction before multiplying by the LR.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = "none",
        update_noise_std: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("Adam beta values must be in [0,1)")
        if weight_decay < 0 or update_noise_std < 0:
            raise ValueError("weight_decay and update_noise_std must be non-negative")
        if weight_decay_mode not in ("none", "zero", "init"):
            raise ValueError(f"Unsupported weight_decay_mode={weight_decay_mode!r}")
        defaults = dict(
            lr=float(lr),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
            weight_decay_mode=weight_decay_mode,
            update_noise_std=float(update_noise_std),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            mode = group["weight_decay_mode"]
            noise_std = group["update_noise_std"]

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("PaperAdam does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if mode == "init":
                        state["initial_parameter"] = parameter.detach().clone()

                if weight_decay > 0 and mode == "zero":
                    parameter.mul_(1.0 - lr * weight_decay)
                elif weight_decay > 0 and mode == "init":
                    initial = state["initial_parameter"]
                    parameter.add_(initial - parameter, alpha=lr * weight_decay)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
                denominator.add_(group["eps"])
                update = exp_avg / bias_correction1 / denominator
                if noise_std:
                    update = update + torch.randn_like(update) * noise_std
                parameter.add_(update, alpha=-lr)

        return loss


def warmup_learning_rate(base_lr: float, step_index: int, warmup_steps: int) -> float:
    """Linear warmup on the first N updates, then a constant learning rate."""
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min((step_index + 1) / warmup_steps, 1.0)


def set_step_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    step_index: int,
    warmup_steps: int,
) -> float:
    lr = warmup_learning_rate(base_lr, step_index, warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def make_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
    weight_decay_mode: str = "zero",
    betas: tuple[float, float] = (0.9, 0.98),
    update_noise_std: float = 0.0,
) -> PaperAdam:
    normalized = name.lower()
    if normalized not in {"adam", "adamw"}:
        raise ValueError("optimizer name must be 'adam' or 'adamw'")
    if normalized == "adam":
        mode: WeightDecayMode = "none"
        decay = 0.0
    else:
        mode = weight_decay_mode if weight_decay > 0 else "none"  # type: ignore[assignment]
        decay = weight_decay
    return PaperAdam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=1e-8,
        weight_decay=decay,
        weight_decay_mode=mode,
        update_noise_std=update_noise_std,
    )
