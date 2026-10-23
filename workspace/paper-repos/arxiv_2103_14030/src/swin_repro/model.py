"""Swin Transformer (Liu et al., 2021, Section 3).

* Patch partition 4x4 + linear embedding; patch merging (2x2 concat -> LN -> Linear 4C->2C) between stages.
* Each pair of blocks uses W-MSA then SW-MSA (windows shifted by M/2 via cyclic shift + masking).
* Relative position bias B in R^{M^2 x M^2} gathered from a table of size (2M-1)^2 per head.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = {
    "swin_debug": dict(embed=24, depths=(2, 2, 2, 2), heads=(1, 2, 3, 4), window=4),
    "swin_tiny": dict(embed=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=7),
    "swin_small": dict(embed=96, depths=(2, 2, 18, 2), heads=(3, 6, 12, 24), window=7),
    "swin_base": dict(embed=128, depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), window=7),
    "swin_large": dict(embed=192, depths=(2, 2, 18, 2), heads=(6, 12, 24, 48), window=7),
}


def window_partition(x: torch.Tensor, M: int) -> torch.Tensor:
    """[B, H, W, C] -> [B * nW, M*M, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // M, M, W // M, M, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, M * M, C)


def window_reverse(w: torch.Tensor, M: int, H: int, W: int) -> torch.Tensor:
    B = w.shape[0] // ((H // M) * (W // M))
    x = w.view(B, H // M, W // M, M, M, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


def relative_position_index(M: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij")).flatten(1)  # [2, M*M]
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (M - 1)  # in [0, 2M-2]
    return rel[..., 0] * (2 * M - 1) + rel[..., 1]  # [M*M, M*M]


def shifted_window_mask(H: int, W: int, M: int, s: int) -> torch.Tensor:
    """Additive mask [nW, M*M, M*M]: -100 where tokens come from different pre-shift regions."""
    img = torch.zeros(1, H, W, 1)
    cnt = 0
    for hs in (slice(0, -M), slice(-M, -s), slice(-s, None)):
        for ws in (slice(0, -M), slice(-M, -s), slice(-s, None)):
            img[:, hs, ws, :] = cnt
            cnt += 1
    mw = window_partition(img, M).squeeze(-1)
    diff = mw[:, None, :] - mw[:, :, None]
    return diff.ne(0).float() * -100.0


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, M):
        super().__init__()
        self.h, self.M, self.scale = heads, M, (dim // heads) ** -0.5
        self.rpb_table = nn.Parameter(torch.zeros((2 * M - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        self.register_buffer("rp_index", relative_position_index(M), persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        q, k, v = self.qkv(x).view(Bw, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rpb_table[self.rp_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.h, N, N) + mask[None, :, None]
            attn = attn.view(-1, self.h, N, N)
        return self.proj((attn.softmax(-1) @ v).transpose(1, 2).reshape(Bw, N, C))


def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) >= p
    return x * keep / (1 - p)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, M, shift, res, dp=0.0):
        super().__init__()
        H, W = res
        if min(H, W) <= M:  # window covers the whole map: no shift needed (as in the reference code)
            M, shift = min(H, W), 0
        self.M, self.shift, self.res, self.dp = M, shift, res, dp
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, M)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.register_buffer("mask", shifted_window_mask(H, W, M, shift) if shift else None, persistent=False)

    def forward(self, x):
        H, W = self.res
        B, L, C = x.shape
        y = self.norm1(x).view(B, H, W, C)
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (1, 2))
        y = window_reverse(self.attn(window_partition(y, self.M), self.mask), self.M, H, W)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (1, 2))
        x = x + drop_path(y.reshape(B, L, C), self.dp, self.training)
        return x + drop_path(self.mlp(self.norm2(x)), self.dp, self.training)


class PatchMerging(nn.Module):
    def __init__(self, dim, res):
        super().__init__()
        self.res = res
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.res
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class SwinTransformer(nn.Module):
    def __init__(self, variant="swin_tiny", img_size=224, n_classes=1000, drop_path_rate=0.2, patch=4):
        super().__init__()
        v = VARIANTS[variant]
        C, M = v["embed"], v["window"]
        self.patch_embed = nn.Conv2d(3, C, patch, patch)
        self.pe_norm = nn.LayerNorm(C)
        res = img_size // patch
        rates = torch.linspace(0, drop_path_rate, sum(v["depths"])).tolist()
        self.layers = nn.ModuleList()
        k = 0
        for i, (dpt, h) in enumerate(zip(v["depths"], v["heads"])):
            dim = C * 2**i
            blocks = [SwinBlock(dim, h, M, 0 if j % 2 == 0 else M // 2, (res, res), rates[k + j]) for j in range(dpt)]
            k += dpt
            if i < 3:
                blocks.append(PatchMerging(dim, (res, res)))
                res //= 2
            self.layers.append(nn.Sequential(*blocks))
        self.norm = nn.LayerNorm(C * 8)
        self.head = nn.Linear(C * 8, n_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pe_norm(self.patch_embed(x).flatten(2).transpose(1, 2))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x).mean(1))
