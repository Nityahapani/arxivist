"""ViT backbone for DINOv2 (Section 4-5): patch embedding, [CLS] token, learned position embeddings
interpolated to the input resolution, LayerScale, stochastic depth, optional SwiGLU FFN (ViT-g), and a
learnable [MASK] token that replaces masked patch embeddings for the iBOT objective."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w12, self.w3 = nn.Linear(d, 2 * hidden), nn.Linear(hidden, d)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0, drop_path=0.0, ls_init=1e-5, swiglu=False):
        super().__init__()
        self.h = heads
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        hid = int(d * mlp_ratio)
        self.mlp = SwiGLU(d, int(hid * 2 / 3)) if swiglu else nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))
        self.ls1, self.ls2 = nn.Parameter(ls_init * torch.ones(d)), nn.Parameter(ls_init * torch.ones(d))
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = (torch.rand(x.size(0), 1, 1, device=x.device) >= self.drop_path).to(x.dtype)
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))
        x = x + self._dp(self.ls1 * a)
        return x + self._dp(self.ls2 * self.mlp(self.norm2(x)))


VARIANTS = {"vit_s14": (384, 12, 6), "vit_b14": (768, 12, 12), "vit_l14": (1024, 24, 16), "vit_g14": (1536, 40, 24)}


class DinoViT(nn.Module):
    def __init__(self, img_size=224, patch=14, dim=384, depth=12, heads=6, mlp_ratio=4.0, drop_path=0.0, swiglu=False):
        super().__init__()
        self.patch, self.dim = patch, dim
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.grid = img_size // patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.grid**2, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, drop_path * i / max(1, depth - 1), swiglu=swiglu) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        nn.init.normal_(self.cls_token, std=1e-6)

    @classmethod
    def from_name(cls, name, img_size=224, drop_path=0.0):
        d, L, h = VARIANTS[name]
        return cls(img_size, 14, d, L, h, 4.0, drop_path, swiglu=name == "vit_g14")

    def interpolate_pos(self, h, w):
        if h == w == self.grid:
            return self.pos_embed
        cls, patch = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch = patch.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=(h, w), mode="bicubic", align_corners=False)
        return torch.cat([cls, patch.permute(0, 2, 3, 1).reshape(1, h * w, -1)], 1)

    def forward(self, x, masks=None):
        """x [B, 3, H, W]; masks [B, n_patches] bool (True = masked). Returns (cls [B, d], patches [B, n, d])."""
        t = self.patch_embed(x)
        B, C, h, w = t.shape
        t = t.flatten(2).transpose(1, 2)
        if masks is not None:
            t = torch.where(masks.unsqueeze(-1), self.mask_token.to(t.dtype).expand_as(t), t)
        t = torch.cat([self.cls_token.expand(B, -1, -1), t], 1) + self.interpolate_pos(h, w)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 0], t[:, 1:]
