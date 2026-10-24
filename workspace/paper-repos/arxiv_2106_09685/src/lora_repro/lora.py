"""LoRA layers (Hu et al., 2021, Section 4.1).

h = W0 x + (alpha / r) * B A x, with W0 frozen, A in R^{r x k}, B in R^{d x r}.
A is initialised with a random Gaussian and B with zeros, so BA = 0 at the start of training.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` and adds a trainable low-rank update.

    Args:
        base: the pretrained linear layer W0 (frozen in place).
        r: rank of the update.
        alpha: scaling constant; the update is scaled by ``alpha / r`` (paper Sec. 4.1).
        dropout: optional dropout on the LoRA input path (used in the GPT-2 experiments).
        init: ``"gaussian"`` (paper text) or ``"kaiming"`` (the released ``loralib`` default).
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0, dropout: float = 0.0,
                 init: str = "gaussian"):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        self.reset_lora(init)

    def reset_lora(self, init: str = "gaussian") -> None:
        if init == "gaussian":
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / self.r)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            raise ValueError(f"unknown init {init!r}")
        nn.init.zeros_(self.lora_B)

    @property
    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold BA into W0 so inference has no extra latency (paper Sec. 4.1)."""
        if not self.merged:
            self.base.weight += self.delta_weight.to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight.to(self.base.weight.dtype)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        return out + F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self) -> str:
        return f"in={self.base.in_features}, out={self.base.out_features}, r={self.r}, alpha={self.alpha}"


def inject_lora(model: nn.Module, target_modules: Iterable[str], r: int, alpha: float,
                dropout: float = 0.0, init: str = "gaussian") -> list[str]:
    """Replace every ``nn.Linear`` whose attribute name is in ``target_modules`` with LoRALinear.

    Returns the dotted names of the replaced modules.
    """
    targets = set(target_modules)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout, init=init))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    if not replaced:
        raise ValueError(f"no nn.Linear named any of {sorted(targets)} found")
    return replaced


def mark_only_lora_as_trainable(model: nn.Module, train_names: Iterable[str] = ()) -> None:
    """Freeze everything except LoRA matrices (plus any parameter whose name contains one of
    ``train_names``, e.g. the task-specific classifier head that the paper also trains)."""
    keep = tuple(train_names)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name or any(k in name for k in keep))


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def merge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def lora_param_count(d_model: int, n_layers: int, r: int, n_matrices: int = 2) -> int:
    """|Theta| = 2 * L_hat_LoRA * d_model * r  for square projections (paper Sec. 7.x / Table 4 note),
    where L_hat_LoRA = n_matrices * n_layers adapted weight matrices."""
    return 2 * n_matrices * n_layers * d_model * r
