"""WideResNet (Zagoruyko & Komodakis, 2016); SAM evaluates WRN-28-10 on CIFAR-10/100."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, cin, cout, stride, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin); self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout); self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.drop = drop
        self.equal = cin == cout and stride == 1
        self.short = None if self.equal else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        y = self.conv1(o)
        y = F.relu(self.bn2(y))
        if self.drop:
            y = F.dropout(y, self.drop, self.training)
        y = self.conv2(y)
        return y + (x if self.equal else self.short(o))


class WideResNet(nn.Module):
    def __init__(self, depth=34, widen=10, n_classes=10, drop=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        w = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        layers, cin = [], w[0]
        for i, stride in enumerate((1, 2, 2)):
            for j in range(n):
                layers.append(WideBasic(cin, w[i + 1], stride if j == 0 else 1, drop))
                cin = w[i + 1]
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.conv1(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
