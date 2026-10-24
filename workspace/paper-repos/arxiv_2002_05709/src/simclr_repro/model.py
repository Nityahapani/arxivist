"""ResNet encoder f(.) and projection head g(.) (Section 2.1, Section 4).

h = f(x) is the representation used downstream; z = g(h) = W2 ReLU(W1 h) is used by the loss.
The CIFAR variant replaces the 7x7 stride-2 stem with a 3x3 stride-1 conv and drops the max
pool (Appendix B.9).
"""
from __future__ import annotations

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


class SimCLR(nn.Module):
    def __init__(self, arch="resnet50", width_mult=1, proj_dim=128, cifar_stem=False, base=64):
        super().__init__()
        self.encoder = ResNet(arch, width_mult, cifar_stem, base)
        d = self.encoder.out_dim
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, proj_dim))

    def forward(self, x):
        h = self.encoder(x)
        return h, self.projector(h)
