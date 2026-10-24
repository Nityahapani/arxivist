"""Feature Pyramid Network (Lin et al., 2017) as used by Mask R-CNN: lateral 1x1 convs, top-down
nearest upsampling, 3x3 output convs; P6 by stride-2 max-pool of P5 (RPN only)."""
from __future__ import annotations

from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, d=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, d, 1) for c in in_channels])
        self.output = nn.ModuleList([nn.Conv2d(d, d, 3, padding=1) for _ in in_channels])

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="nearest")
        outs = [o(x) for o, x in zip(self.output, lat)]
        return outs + [F.max_pool2d(outs[-1], 1, 2)]  # P2..P6
