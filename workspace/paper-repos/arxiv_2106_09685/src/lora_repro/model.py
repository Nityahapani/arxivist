"""Small Transformer with separately named q/k/v/o projections so LoRA can target W_q and W_v,
as in the paper's main configuration (Section 4.2 / 7.1)."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh, self.causal = n_heads, d_model // n_heads, causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = (p(x).view(B, T, self.h, self.dh).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if self.causal:
            att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        if pad_mask is not None:  # [B, T] True for padding
            att = att.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyTransformer(nn.Module):
    """Pre-LN Transformer usable as an encoder classifier (RoBERTa-like) or causal LM (GPT-like)."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 max_len: int = 128, n_classes: int | None = None, causal: bool = False):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, causal) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.causal = causal
        self.head = nn.Linear(d_model, n_classes if n_classes else vocab_size)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.tok(ids) + self.pos(pos)
        for b in self.blocks:
            x = b(x, pad_mask)
        x = self.ln_f(x)
        if self.causal:
            return self.head(x)
        return self.head(x[:, 0])  # first-token ([CLS]-style) pooling
