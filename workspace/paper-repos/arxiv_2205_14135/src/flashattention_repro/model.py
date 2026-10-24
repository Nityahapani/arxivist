"""GPT-style causal LM whose attention can be FlashAttention (reference) or standard attention."""
from __future__ import annotations

import torch
from torch import nn

from .flash import flash_attention, standard_attention


class Block(nn.Module):
    def __init__(self, d, h, ff, impl, br, bc):
        super().__init__()
        self.h, self.impl, self.br, self.bc = h, impl, br, bc
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = flash_attention(q, k, v, self.br, self.bc, True) if self.impl == "flash" else standard_attention(q, k, v, True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=128, n_layers=2, n_heads=4, ff=512, max_len=256, impl="flash", br=16, bc=16):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ff, impl, br, bc) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))
