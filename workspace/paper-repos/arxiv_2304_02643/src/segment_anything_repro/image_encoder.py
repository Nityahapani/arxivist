"""Image encoder (Section 3, Appendix A): an MAE-pretrained ViT adapted to high resolution with
14x14 windowed attention and four equally spaced global-attention blocks, followed by a neck
(1x1 conv -> LayerNorm2d -> 3x3 conv -> LayerNorm2d) producing a 256-channel, 16x-downscaled embedding.
ViT-H: 1024x1024 input -> 64x64x256 embedding.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = {
    "vit_b": dict(dim=768, depth=12, heads=12, global_attn=(2, 5, 8, 11)),
    "vit_l": dict(dim=1024, depth=24, heads=16, global_attn=(5, 11, 17, 23)),
    "vit_h": dict(dim=1280, depth=32, heads=16, global_attn=(7, 15, 23, 31)),
}


def equally_spaced_global(depth: int, n: int = 4) -> tuple[int, ...]:
    step = depth // n
    return tuple(step * (i + 1) - 1 for i in range(n))


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


def window_partition(x, ws):
    """[B, H, W, C] -> ([B * nW, ws, ws, C], (Hp, Wp)) with zero padding to a multiple of ws."""
    B, H, W, C = x.shape
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, (Hp, Wp)


def window_unpartition(win, ws, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = win.shape[0] // (Hp * Wp // ws // ws)
    x = win.view(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W].contiguous()


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x):  # x: [B, H, W, C]
        B, H, W, C = x.shape
        q, k, v = self.qkv(x).view(B, H * W, 3, self.h, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, H, W, C)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim, heads, window_size=0, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.LayerNorm(dim), Attention(dim, heads), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.ws = window_size

    def forward(self, x):
        h = self.n1(x)
        if self.ws > 0:
            H, W = h.shape[1:3]
            h, pad = window_partition(h, self.ws)
            h = window_unpartition(self.attn(h), self.ws, pad, (H, W))
        else:
            h = self.attn(h)
        x = x + h
        return x + self.mlp(self.n2(x))


class ImageEncoderViT(nn.Module):
    def __init__(self, img_size=1024, patch=16, dim=1280, depth=32, heads=16, window_size=14, global_attn=None, out_chans=256):
        super().__init__()
        global_attn = equally_spaced_global(depth) if global_attn is None else tuple(global_attn)
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch, img_size // patch, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, 0 if i in global_attn else window_size) for i in range(depth)])
        self.global_attn = global_attn
        self.neck = nn.Sequential(nn.Conv2d(dim, out_chans, 1, bias=False), LayerNorm2d(out_chans),
                                  nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False), LayerNorm2d(out_chans))

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        return self.neck(x.permute(0, 3, 1, 2))
