"""ConvNeXt (Liu et al., 2022, Section 2 / Fig. 4).

Block: 7x7 depthwise conv -> LayerNorm -> Linear(4C) -> GELU -> Linear(C) -> layer scale -> drop path.
Stem: 4x4 stride-4 "patchify" conv + LN. Downsampling: LN + 2x2 stride-2 conv between stages.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = {
    "convnext_atto_debug": dict(depths=(1, 1, 1, 1), dims=(16, 32, 64, 128)),
    "convnext_tiny": dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)),
    "convnext_small": dict(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768)),
    "convnext_base": dict(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)),
    "convnext_large": dict(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536)),
    "convnext_xlarge": dict(depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048)),
}


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2], self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


def drop_path(x, p: float, training: bool):
    if p == 0.0 or not training:
        return x
    keep = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) >= p
    return x * keep / (1 - p)


class Block(nn.Module):
    def __init__(self, dim, drop_path_rate=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.dp = drop_path_rate

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        if self.gamma is not None:
            y = self.gamma * y
        return x + drop_path(y.permute(0, 3, 1, 2), self.dp, self.training)


class ConvNeXt(nn.Module):
    def __init__(self, variant="convnext_tiny", in_ch=3, n_classes=1000, drop_path_rate=0.0, layer_scale_init=1e-6, head_init_scale=1.0):
        super().__init__()
        depths, dims = VARIANTS[variant]["depths"], VARIANTS[variant]["dims"]
        self.downsample = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, dims[0], 4, 4), LayerNorm2d(dims[0]))])
        for i in range(3):
            self.downsample.append(nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, 2)))
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()  # linearly increasing stochastic depth
        self.stages, k = nn.ModuleList(), 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], rates[k + j], layer_scale_init) for j in range(depths[i])]))
            k += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], n_classes)
        self.apply(self._init)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        for ds, st in zip(self.downsample, self.stages):
            x = st(ds(x))
        return self.norm(x.mean((-2, -1)))

    def forward(self, x):
        return self.head(self.forward_features(x))
