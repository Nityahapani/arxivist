"""RoFormer: a BERT-style Transformer encoder whose self-attention applies RoPE to queries and keys
(Section 3.4), with an optional linear-attention variant (Section 3.3, Eq. 19).

`pos` selects the position method: "rope" (RoFormer), "sinusoidal" (absolute, Vaswani et al.), or
"none" — useful for the baselines.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .rope import apply_rope


def sinusoidal(T, d, device=None):
    pos = torch.arange(T, device=device).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def linear_attention_rope(q, k, v, use_rope=True, eps=1e-6):
    """Eq. 19: Attention_m = sum_n (R_m phi(q_m))^T (R_n phi(k_n)) v_n / sum_n phi(q_m)^T phi(k_n).

    RoPE is applied to the feature-mapped queries/keys in the numerator only, keeping the
    normaliser positive. phi = elu + 1. Non-causal (encoder) form.
    """
    fq, fk = F.elu(q) + 1, F.elu(k) + 1
    denom = torch.einsum("bhtd,bhd->bht", fq, fk.sum(2)).clamp_min(eps)
    rq, rk = (apply_rope(fq), apply_rope(fk)) if use_rope else (fq, fk)
    num = torch.einsum("bhtd,bhde->bhte", rq, torch.einsum("bhtd,bhte->bhde", rk, v))
    return num / denom[..., None]


class Attention(nn.Module):
    def __init__(self, d, h, pos="rope", linear=False):
        super().__init__()
        self.h, self.pos, self.linear = h, pos, linear
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, pad_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.linear:
            y = linear_attention_rope(q, k, v, use_rope=self.pos == "rope")
        else:
            if self.pos == "rope":
                q, k = apply_rope(q), apply_rope(k)
            mask = None if pad_mask is None else pad_mask[:, None, None, :]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h, ff, pos, linear, dropout):
        super().__init__()
        self.attn = Attention(d, h, pos, linear)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):  # post-LN as in BERT
        x = self.ln1(x + self.drop(self.attn(x, pad_mask)))
        return self.ln2(x + self.drop(self.ff(x)))


class RoFormer(nn.Module):
    def __init__(self, vocab, d=768, n_layers=12, n_heads=12, ff=3072, pos="rope", linear=False, dropout=0.1):
        super().__init__()
        self.pos, self.d = pos, d
        self.emb = nn.Embedding(vocab, d)
        self.ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ff, pos, linear, dropout) for _ in range(n_layers)])
        self.mlm = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, vocab))

    def forward(self, ids, pad_mask=None):
        x = self.emb(ids)
        if self.pos == "sinusoidal":
            x = x + sinusoidal(ids.size(1), self.d, ids.device)
        x = self.ln(x)
        for b in self.blocks:
            x = b(x, pad_mask)
        return self.mlm(x)
