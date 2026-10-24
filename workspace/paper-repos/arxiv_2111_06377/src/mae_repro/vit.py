"""Vision Transformer building blocks and fixed 2-D sine-cosine position embeddings."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d(dim: int, grid: int, cls_token: bool = False) -> np.ndarray:
    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float64), np.arange(grid, dtype=np.float64), indexing="ij")
    emb = np.concatenate([sincos_1d(dim // 2, gw), sincos_1d(dim // 2, gh)], axis=1)  # [grid*grid, dim]
    if cls_token:
        emb = np.concatenate([np.zeros([1, dim]), emb], axis=0)
    return emb


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch, in_ch, dim):
        super().__init__()
        self.grid = img_size // patch
        self.num_patches = self.grid**2
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
