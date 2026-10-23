"""DETR (Section 3.2, Fig. 2 and Fig. 10).

CNN backbone -> 1x1 conv to d -> Transformer encoder (spatial positional encodings added to Q/K in every
layer) -> Transformer decoder over N learned object queries (query positional encodings added in every
layer) -> shared FFN heads after each decoder layer: class (C + 1 with no-object) and box (3-layer MLP,
sigmoid, normalised cx, cy, w, h).
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import Backbone, freeze_bn


def sine_position_encoding(h, w, d, temperature=10000.0, device=None):
    """2D sine encoding: d/2 channels for y, d/2 for x, normalised coordinates scaled by 2*pi."""
    npf = d // 2
    y = (torch.arange(1, h + 1, device=device).float() / h * 2 * math.pi)[:, None].expand(h, w)
    x = (torch.arange(1, w + 1, device=device).float() / w * 2 * math.pi)[None].expand(h, w)
    dim_t = temperature ** (2 * (torch.arange(npf, device=device) // 2) / npf)
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], -1).flatten(-2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], -1).flatten(-2)
    return torch.cat([py, px], -1).flatten(0, 1)  # [h*w, d]


class EncoderLayer(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(p), nn.Linear(ff, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, x, pos):
        q = k = x + pos
        x = self.n1(x + self.drop(self.attn(q, k, x, need_weights=False)[0]))
        return self.n2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(p), nn.Linear(ff, d))
        self.n1, self.n2, self.n3, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, tgt, memory, pos, query_pos):
        q = k = tgt + query_pos
        tgt = self.n1(tgt + self.drop(self.self_attn(q, k, tgt, need_weights=False)[0]))
        tgt = self.n2(tgt + self.drop(self.cross_attn(tgt + query_pos, memory + pos, memory, need_weights=False)[0]))
        return self.n3(tgt + self.drop(self.ff(tgt)))


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n):
        super().__init__()
        dims = [d_in] + [d_hidden] * (n - 1)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:] + [d_out])])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < len(self.layers) - 1 else l(x)
        return x


class DETR(nn.Module):
    def __init__(self, n_classes, n_queries=100, d=256, heads=8, enc_layers=6, dec_layers=6, ff=2048, dropout=0.1,
                 arch="resnet50", base_width=64):
        super().__init__()
        self.backbone = freeze_bn(Backbone(arch, 1, False, base_width))
        self.input_proj = nn.Conv2d(self.backbone.out_dim, d, 1)
        self.encoder = nn.ModuleList([EncoderLayer(d, heads, ff, dropout) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d, heads, ff, dropout) for _ in range(dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.query_embed = nn.Embedding(n_queries, d)
        self.class_head = nn.Linear(d, n_classes + 1)
        self.box_head = MLP(d, d, 4, 3)
        self.d = d

    def train(self, mode=True):
        super().train(mode)
        freeze_bn(self.backbone)
        return self

    def forward(self, images):
        f = self.input_proj(self.backbone(images))
        B, d, h, w = f.shape
        src = f.flatten(2).transpose(1, 2)
        pos = sine_position_encoding(h, w, d, device=f.device).unsqueeze(0).expand(B, -1, -1)
        for layer in self.encoder:
            src = layer(src, pos)
        qpos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(qpos)
        hs = []
        for layer in self.decoder:
            tgt = layer(tgt, src, pos, qpos)
            hs.append(self.dec_norm(tgt))
        hs = torch.stack(hs)  # [L, B, N, d]
        return {"logits": self.class_head(hs), "boxes": self.box_head(hs).sigmoid()}
