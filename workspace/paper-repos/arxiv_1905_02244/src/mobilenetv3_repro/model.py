"""MobileNetV3-Large / -Small (Howard et al., 2019), Tables 1-2 and Section 5.

bneck = 1x1 expansion -> kxk depthwise -> optional SE (hard-sigmoid gate) -> 1x1 linear projection,
with ReLU or h-swish nonlinearity; efficient last stage (Section 5.1) moves the 1x1 expansion after
global pooling.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def h_sigmoid(x):
    return F.relu6(x + 3.0) / 6.0


def h_swish(x):
    """h-swish(x) = x * ReLU6(x + 3) / 6  (Section 5.2, Eq. 2)."""
    return x * h_sigmoid(x)


class HSwish(nn.Module):
    def forward(self, x):
        return h_swish(x)


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new < 0.9 * v:
        new += divisor
    return new


# (kernel, expansion size, out, SE, NL, stride)
LARGE = [
    (3, 16, 16, False, "RE", 1), (3, 64, 24, False, "RE", 2), (3, 72, 24, False, "RE", 1),
    (5, 72, 40, True, "RE", 2), (5, 120, 40, True, "RE", 1), (5, 120, 40, True, "RE", 1),
    (3, 240, 80, False, "HS", 2), (3, 200, 80, False, "HS", 1), (3, 184, 80, False, "HS", 1),
    (3, 184, 80, False, "HS", 1), (3, 480, 112, True, "HS", 1), (3, 672, 112, True, "HS", 1),
    (5, 672, 160, True, "HS", 2), (5, 960, 160, True, "HS", 1), (5, 960, 160, True, "HS", 1),
]
SMALL = [
    (3, 16, 16, True, "RE", 2), (3, 72, 24, False, "RE", 2), (3, 88, 24, False, "RE", 1),
    (5, 96, 40, True, "HS", 2), (5, 240, 40, True, "HS", 1), (5, 240, 40, True, "HS", 1),
    (5, 120, 48, True, "HS", 1), (5, 144, 48, True, "HS", 1), (5, 288, 96, True, "HS", 2),
    (5, 576, 96, True, "HS", 1), (5, 576, 96, True, "HS", 1),
]
CONFIGS = {"large": (LARGE, 960, 1280), "small": (SMALL, 576, 1024)}


class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        r = make_divisible(c // 4)
        self.fc1, self.fc2 = nn.Conv2d(c, r, 1), nn.Conv2d(r, c, 1)

    def forward(self, x):
        s = F.relu(self.fc1(x.mean((2, 3), keepdim=True)))
        return x * h_sigmoid(self.fc2(s))


class Bneck(nn.Module):
    def __init__(self, cin, k, exp, cout, se, nl, stride):
        super().__init__()
        act = nn.ReLU if nl == "RE" else HSwish
        layers = []
        if exp != cin:
            layers += [nn.Conv2d(cin, exp, 1, bias=False), nn.BatchNorm2d(exp, eps=1e-3, momentum=0.01), act()]
        layers += [nn.Conv2d(exp, exp, k, stride, k // 2, groups=exp, bias=False), nn.BatchNorm2d(exp, eps=1e-3, momentum=0.01), act()]
        if se:
            layers.append(SE(exp))
        layers += [nn.Conv2d(exp, cout, 1, bias=False), nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)]
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return x + y if self.residual else y


class MobileNetV3(nn.Module):
    def __init__(self, mode="large", n_classes=1000, width_mult=1.0, dropout=0.2):
        super().__init__()
        cfg, last_conv, last_fc = CONFIGS[mode]
        c = make_divisible(16 * width_mult)
        layers = [nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c, eps=1e-3, momentum=0.01), HSwish()]
        for k, exp, out, se, nl, s in cfg:
            e, o = make_divisible(exp * width_mult), make_divisible(out * width_mult)
            layers.append(Bneck(c, k, e, o, se, nl, s))
            c = o
        lc = make_divisible(last_conv * width_mult)
        layers += [nn.Conv2d(c, lc, 1, bias=False), nn.BatchNorm2d(lc, eps=1e-3, momentum=0.01), HSwish()]
        self.features = nn.Sequential(*layers)
        # Efficient last stage: pool first, then 1x1 conv (as Linear) without BN
        self.classifier = nn.Sequential(nn.Linear(lc, last_fc), HSwish(), nn.Dropout(dropout), nn.Linear(last_fc, n_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(self.features(x), 1), 1))
