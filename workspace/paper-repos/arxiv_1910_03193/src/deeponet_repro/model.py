"""DeepONet architectures (Section 3, Fig. 1) and the FNN baseline.

G(u)(y) ≈ sum_{k=1}^{p} b_k(u(x_1), ..., u(x_m)) · t_k(y) + b_0
* stacked:   p independent branch nets, each producing one scalar b_k.
* unstacked: one branch net producing all p coefficients.
"""
from __future__ import annotations

import torch
from torch import nn

ACT = {"relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(sizes: list[int], act: str = "relu", last_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(ACT[act]())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, m: int, y_dim: int = 1, p: int = 40, branch_hidden=(40,), trunk_hidden=(40, 40),
                 act: str = "relu", stacked: bool = False, use_bias: bool = True):
        super().__init__()
        self.p, self.stacked = p, stacked
        if stacked:
            self.branch = nn.ModuleList([mlp([m, *branch_hidden, 1], act) for _ in range(p)])
        else:
            self.branch = mlp([m, *branch_hidden, p], act)
        # Trunk applies the activation in its last layer as well (paper Sec. 3: t_k are features).
        self.trunk = mlp([y_dim, *trunk_hidden, p], act, last_act=True)
        self.b0 = nn.Parameter(torch.zeros(1)) if use_bias else None

    def branch_out(self, u: torch.Tensor) -> torch.Tensor:
        if self.stacked:
            return torch.cat([b(u) for b in self.branch], dim=-1)
        return self.branch(u)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = (self.branch_out(u) * self.trunk(y)).sum(-1, keepdim=True)
        return out + self.b0 if self.b0 is not None else out


class FNNBaseline(nn.Module):
    """Fully connected network on the concatenated input [u(x_1..m), y] (paper's FNN baseline)."""

    def __init__(self, m: int, y_dim: int = 1, hidden=(100, 100, 100), act: str = "relu"):
        super().__init__()
        self.net = mlp([m + y_dim, *hidden, 1], act)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u, y], dim=-1))


def build(cfg: dict, m: int) -> nn.Module:
    kind = cfg.get("kind", "deeponet_unstacked")
    if kind == "fnn":
        return FNNBaseline(m, hidden=tuple(cfg.get("hidden", (100, 100, 100))), act=cfg.get("act", "relu"))
    return DeepONet(m, p=cfg.get("p", 40), branch_hidden=tuple(cfg.get("branch_hidden", (40,))),
                    trunk_hidden=tuple(cfg.get("trunk_hidden", (40, 40))), act=cfg.get("act", "relu"),
                    stacked=kind == "deeponet_stacked", use_bias=cfg.get("bias", True))
