"""ResNet backbone returning C2-C5 feature maps (strides 4, 8, 16, 32) for FPN (Section 3, 'Network
Architecture'). BatchNorm is frozen as in the paper's fine-tuning from ImageNet."""
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


class MultiScaleResNet(ResNet):
    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # [C2, C3, C4, C5]

    def __init__(self, arch="resnet50", width_mult=1, cifar_stem=False, base=64):
        super().__init__(arch, width_mult, cifar_stem, base)
        block, _ = DEPTHS[arch]
        self.channels = [base * width_mult * 2**i * block.expansion for i in range(4)]


def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    return module
