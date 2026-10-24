"""BYOL networks (Section 3.1, Section 3.3).

Online network: encoder f_theta (ResNet), projector g_theta, predictor q_theta.
Target network: encoder f_xi and projector g_xi, an exponential moving average of the online weights.
Projector and predictor are MLPs: Linear(d, 4096) -> BN -> ReLU -> Linear(4096, 256).
The CIFAR stem (3x3 stride-1 conv, no max pool) is a convenience for small images.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.b2(self.c2(torch.relu(self.b1(self.c1(x))))) + self.sc(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, cmid, stride):
        super().__init__()
        cout = cmid * 4
        self.net = nn.Sequential(
            nn.Conv2d(cin, cmid, 1, bias=False), nn.BatchNorm2d(cmid), nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cmid, 3, stride, 1, bias=False), nn.BatchNorm2d(cmid), nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.net(x) + self.sc(x))


DEPTHS = {"resnet18": (BasicBlock, [2, 2, 2, 2]), "resnet50": (Bottleneck, [3, 4, 6, 3])}


class ResNet(nn.Module):
    def __init__(self, arch="resnet50", width_mult=1, cifar_stem=False, base=64):
        super().__init__()
        block, layers = DEPTHS[arch]
        w = base * width_mult
        if cifar_stem:
            self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, w, 7, 2, 3, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                                      nn.MaxPool2d(3, 2, 1))
        cin, stages = w, []
        for i, n in enumerate(layers):
            cmid = w * 2**i
            blocks = []
            for j in range(n):
                blocks.append(block(cin, cmid, 2 if (j == 0 and i > 0) else 1))
                cin = cmid * block.expansion
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.out_dim = cin

    def forward(self, x):
        return torch.flatten(nn.functional.adaptive_avg_pool2d(self.stages(self.stem(x)), 1), 1)


def mlp(d_in, hidden=4096, d_out=256):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d_out))


class BYOL(nn.Module):
    def __init__(self, arch="resnet50", width_mult=1, cifar_stem=False, base=64, hidden=4096, proj_dim=256):
        super().__init__()
        self.encoder = ResNet(arch, width_mult, cifar_stem, base)
        self.projector = mlp(self.encoder.out_dim, hidden, proj_dim)
        self.predictor = mlp(proj_dim, hidden, proj_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_parameters():
            p.requires_grad_(False)

    def online_parameters(self):
        return [*self.encoder.parameters(), *self.projector.parameters(), *self.predictor.parameters()]

    def target_parameters(self):
        return [*self.target_encoder.parameters(), *self.target_projector.parameters()]

    def online(self, x):
        return self.predictor(self.projector(self.encoder(x)))

    @torch.no_grad()
    def target(self, x):
        return self.target_projector(self.target_encoder(x))

    @torch.no_grad()
    def update_target(self, tau: float):
        """xi <- tau * xi + (1 - tau) * theta  (Eq. 1). BN running statistics are copied."""
        for online, target in ((self.encoder, self.target_encoder), (self.projector, self.target_projector)):
            for pt, po in zip(target.parameters(), online.parameters()):
                pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
            for bt, bo in zip(target.buffers(), online.buffers()):
                bt.copy_(bo)
