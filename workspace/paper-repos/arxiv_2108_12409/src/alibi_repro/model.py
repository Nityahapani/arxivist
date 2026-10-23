"""Causal Transformer LM with interchangeable position methods: ALiBi, sinusoidal, rotary, learned."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .alibi import alibi_bias


def sinusoidal(T: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def rotary(x: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to [B, H, T, Dh] (used for the rotary baseline)."""
    B, H, T, D = x.shape
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    ang = torch.arange(T, device=x.device).float()[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, h, pos):
        super().__init__()
        self.h, self.pos = h, pos
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.pos == "rotary":
            q, k = rotary(q), rotary(k)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        mask = torch.zeros(T, T, device=x.device).masked_fill(~causal, float("-inf"))
        if self.pos == "alibi":
            mask = mask + alibi_bias(self.h, T, device=x.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h, ff, pos, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, pos)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerLM(nn.Module):
    def __init__(self, vocab, d=128, n_layers=2, n_heads=4, ff=512, pos="alibi", max_len=None, dropout=0.0):
        super().__init__()
        assert pos in ("alibi", "sinusoidal", "rotary", "learned")
        self.pos, self.d = pos, d
        self.emb = nn.Embedding(vocab, d)
        self.learned = nn.Embedding(max_len, d) if pos == "learned" else None
        self.blocks = nn.ModuleList([Block(d, n_heads, ff, pos, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied input/output embeddings

    def forward(self, ids):
        T = ids.size(1)
        x = self.emb(ids) * math.sqrt(self.d)
        if self.pos == "sinusoidal":
            x = x + sinusoidal(T, self.d, ids.device)
        elif self.pos == "learned":
            x = x + self.learned(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))
