"""Time-conditional UNet with cross-attention conditioning (Section 3.3, Fig. 3).

epsilon_theta(z_t, t, tau_theta(y)): ResBlocks with timestep embeddings; at selected resolutions a
SpatialTransformer whose cross-attention uses Q = W_Q phi(z_t), K = W_K tau(y), V = W_V tau(y)
(Eq. 4-5). With context=None the cross-attention reduces to self-attention (unconditional).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .autoencoder import Norm


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


class TResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1, self.c1 = Norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, cout)
        self.n2, self.c2 = Norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x))) + self.t(F.silu(emb))[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class CrossAttention(nn.Module):
    def __init__(self, d, d_ctx=None, heads=4):
        super().__init__()
        d_ctx = d_ctx or d
        self.h = heads
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d_ctx, d, bias=False), nn.Linear(d_ctx, d, bias=False)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.h, -1).transpose(1, 2)
        k = self.k(ctx).view(B, ctx.size(1), self.h, -1).transpose(1, 2)
        v = self.v(ctx).view(B, ctx.size(1), self.h, -1).transpose(1, 2)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class SpatialTransformer(nn.Module):
    def __init__(self, c, d_ctx, heads):
        super().__init__()
        self.norm, self.proj_in, self.proj_out = Norm(c), nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 1)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(c), nn.LayerNorm(c), nn.LayerNorm(c)
        self.self_attn = CrossAttention(c, None, heads)
        self.cross_attn = CrossAttention(c, d_ctx, heads)
        self.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        self.has_ctx = d_ctx is not None

    def forward(self, x, ctx=None):
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)
        h = h + self.self_attn(self.ln1(h))
        h = h + self.cross_attn(self.ln2(h), ctx if self.has_ctx else None)
        h = h + self.ff(self.ln3(h))
        return x + self.proj_out(h.transpose(1, 2).view(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, in_ch=4, ch=128, mult=(1, 2, 4), attn_levels=(1, 2), n_res=2, heads=4, d_ctx=None, n_classes=None):
        super().__init__()
        temb = 4 * ch
        self.ch = ch
        self.time = nn.Sequential(nn.Linear(ch, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.class_emb = nn.Embedding(n_classes, temb) if n_classes else None
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down, chans, c = nn.ModuleList(), [ch], ch
        for lvl, m in enumerate(mult):
            for _ in range(n_res):
                blk = nn.ModuleList([TResBlock(c, ch * m, temb)]); c = ch * m
                if lvl in attn_levels:
                    blk.append(SpatialTransformer(c, d_ctx, heads))
                self.down.append(blk); chans.append(c)
            if lvl < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)])); chans.append(c)
        self.mid = nn.ModuleList([TResBlock(c, c, temb), SpatialTransformer(c, d_ctx, heads), TResBlock(c, c, temb)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(n_res + 1):
                blk = nn.ModuleList([TResBlock(c + chans.pop(), ch * m, temb)]); c = ch * m
                if lvl in attn_levels:
                    blk.append(SpatialTransformer(c, d_ctx, heads))
                if lvl > 0 and i == n_res:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv2d(c, c, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(Norm(c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))

    @staticmethod
    def _run_block(blk, h, emb, ctx):
        for layer in blk:
            if isinstance(layer, TResBlock):
                h = layer(h, emb)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, ctx)
            else:
                h = layer(h)
        return h

    def forward(self, x, t, context=None, y=None):
        emb = self.time(timestep_embedding(t, self.ch))
        if self.class_emb is not None and y is not None:
            emb = emb + self.class_emb(y)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = self._run_block(blk, h, emb, context)
            hs.append(h)
        h = self._run_block(self.mid, h, emb, context)
        for blk in self.up:
            h = self._run_block(blk, torch.cat([h, hs.pop()], 1), emb, context)
        return self.out(h)
