"""EfficientNet-B0 baseline (Table 1) and compound scaling to B1-B7 (Section 3.3, Eq. 3)."""
from __future__ import annotations

import math

import torch
from torch import nn

# (expand_ratio, kernel, stride, repeats, in_ch, out_ch) — Table 1, stages 2-8
B0_BLOCKS = [
    (1, 3, 1, 1, 32, 16), (6, 3, 2, 2, 16, 24), (6, 5, 2, 2, 24, 40), (6, 3, 2, 3, 40, 80),
    (6, 5, 1, 3, 80, 112), (6, 5, 2, 4, 112, 192), (6, 3, 1, 1, 192, 320),
]
# (width_mult, depth_mult, resolution, dropout)
SCALING = {
    "b0": (1.0, 1.0, 224, 0.2), "b1": (1.0, 1.1, 240, 0.2), "b2": (1.1, 1.2, 260, 0.3), "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4), "b5": (1.6, 2.2, 456, 0.4), "b6": (1.8, 2.6, 528, 0.5), "b7": (2.0, 3.1, 600, 0.5),
}
ALPHA, BETA, GAMMA = 1.2, 1.1, 1.15  # grid-searched with phi = 1 under alpha * beta^2 * gamma^2 ≈ 2


def compound_coefficients(phi: float) -> tuple[float, float, float]:
    """depth d = alpha^phi, width w = beta^phi, resolution r = gamma^phi (Eq. 3)."""
    return ALPHA**phi, BETA**phi, GAMMA**phi


def round_filters(c: int, width_mult: float, divisor: int = 8) -> int:
    c *= width_mult
    new = max(divisor, int(c + divisor / 2) // divisor * divisor)
    if new < 0.9 * c:
        new += divisor
    return int(new)


def round_repeats(r: int, depth_mult: float) -> int:
    return int(math.ceil(depth_mult * r))


class SqueezeExcite(nn.Module):
    def __init__(self, c, c_reduced):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, c_reduced, 1), nn.Conv2d(c_reduced, c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(nn.functional.silu(self.fc1(s))))


class MBConv(nn.Module):
    def __init__(self, cin, cout, expand, k, stride, se_ratio=0.25, drop_connect=0.0, bn_momentum=0.01):
        super().__init__()
        mid = cin * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid, momentum=bn_momentum, eps=1e-3), nn.SiLU()]
        layers += [nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False), nn.BatchNorm2d(mid, momentum=bn_momentum, eps=1e-3), nn.SiLU(),
                   SqueezeExcite(mid, max(1, int(cin * se_ratio))),
                   nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout, momentum=bn_momentum, eps=1e-3)]
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout
        self.drop_connect = drop_connect

    def forward(self, x):
        y = self.block(x)
        if not self.residual:
            return y
        if self.training and self.drop_connect > 0:
            keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_connect
            y = y * keep / (1 - self.drop_connect)
        return x + y


class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout=0.2, n_classes=1000, drop_connect_rate=0.2):
        super().__init__()
        stem = round_filters(32, width_mult)
        layers = [nn.Conv2d(3, stem, 3, 2, 1, bias=False), nn.BatchNorm2d(stem, momentum=0.01, eps=1e-3), nn.SiLU()]
        total = sum(round_repeats(b[3], depth_mult) for b in B0_BLOCKS)
        k = 0
        for e, ks, s, r, cin, cout in B0_BLOCKS:
            cin, cout = round_filters(cin, width_mult), round_filters(cout, width_mult)
            for j in range(round_repeats(r, depth_mult)):
                layers.append(MBConv(cin if j == 0 else cout, cout, e, ks, s if j == 0 else 1, drop_connect=drop_connect_rate * k / total))
                k += 1
        head = round_filters(1280, width_mult)
        layers += [nn.Conv2d(round_filters(320, width_mult), head, 1, bias=False), nn.BatchNorm2d(head, momentum=0.01, eps=1e-3), nn.SiLU()]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(head, n_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(self.features(x), 1), 1))


def efficientnet(name: str, n_classes: int = 1000, **kw) -> EfficientNet:
    w, d, _, p = SCALING[name]
    return EfficientNet(w, d, p, n_classes, **kw)
