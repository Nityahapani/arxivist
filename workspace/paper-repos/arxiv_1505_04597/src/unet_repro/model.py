"""Original U-Net (Ronneberger et al., 2015, Section 2, Fig. 1).

Unpadded 3x3 convolutions, 2x2 max-pool, 2x2 up-convolutions halving channels, crop-and-concatenate
skip connections, final 1x1 conv to the class map. A 572x572 input yields a 388x388 output.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def double_conv(cin, cout, padding=0):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=padding), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=padding), nn.ReLU(inplace=True))


def center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    t, l = (H - h) // 2, (W - w) // 2
    return x[..., t:t + h, l:l + w]


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=2, base=64, depth=4, dropout=0.5, padding=0):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([double_conv(in_ch if i == 0 else chs[i - 1], chs[i], padding) for i in range(depth)])
        self.bottom = double_conv(chs[depth - 1], chs[depth], padding)
        self.drop = nn.Dropout(dropout)  # "drop-out layers at the end of the contracting path" (Section 3.1)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, 2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([double_conv(chs[i + 1], chs[i], padding) for i in reversed(range(depth))])
        self.out = nn.Conv2d(chs[0], n_classes, 1)
        self.pool = nn.MaxPool2d(2)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        """Gaussian init with std sqrt(2/N), N = k*k*C_in (Section 3)."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            k = m.kernel_size[0] * m.kernel_size[1]
            cin = m.in_channels
            nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / (k * cin)))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.drop(self.bottom(x))
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([center_crop(s, *x.shape[-2:]), x], 1))
        return self.out(x)


def output_size(input_size: int, depth: int = 4) -> int:
    """Spatial output size for valid convolutions (572 -> 388 for depth 4)."""
    s = input_size
    for _ in range(depth):
        s = (s - 4) // 2
    s -= 4
    for _ in range(depth):
        s = s * 2 - 4
    return s
