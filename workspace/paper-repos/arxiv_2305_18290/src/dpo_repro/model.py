"""Small causal Transformer LM (GPT-style, learned positions) used as policy and reference in tests
and smoke runs. Paper-scale runs plug in Hugging Face causal LMs (GPT-2 large, GPT-J, Pythia-2.8B)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=64, n_layers=2, n_heads=4, ff=256, max_len=64):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ff) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


@torch.no_grad()
def sample(model, prompts: torch.Tensor, n_new: int, temperature: float = 1.0, g: torch.Generator | None = None):
    ids = prompts
    for _ in range(n_new):
        logits = model(ids)[:, -1] / temperature
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
        ids = torch.cat([ids, nxt], 1)
    return ids
