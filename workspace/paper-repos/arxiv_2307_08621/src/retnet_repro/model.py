"""RetNet language model (Section 2.2-2.3).

MSR(X) = (swish(X W_G) ⊙ GroupNorm_h(head_1..head_h)) W_O, value/gate dimension 2d.
Block: Y = MSR(LN(X)) + X;  X' = FFN(LN(Y)) + Y  with FFN(X) = gelu(X W_1) W_2.
Supports parallel training, chunkwise training, and O(1)-state recurrent decoding.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .retention import head_gammas, retention_chunkwise, retention_parallel, retention_recurrent_step, rotate


class MultiScaleRetention(nn.Module):
    def __init__(self, d, h, v_factor=2):
        super().__init__()
        self.h, self.dk, self.dv = h, d // h, v_factor * d // h
        self.wq, self.wk = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.wv, self.wg = nn.Linear(d, v_factor * d, bias=False), nn.Linear(d, v_factor * d, bias=False)
        self.wo = nn.Linear(v_factor * d, d, bias=False)
        self.gn = nn.GroupNorm(h, v_factor * d, affine=False)
        self.register_buffer("gamma", head_gammas(h).float(), persistent=False)

    def _qkv(self, x, offset=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.dk).transpose(1, 2) * self.dk ** -0.5
        v = self.wv(x).view(B, T, self.h, self.dv).transpose(1, 2)
        return rotate(q, offset), rotate(k, offset), v

    def _out(self, x, y):
        B, H, T, dv = y.shape
        y = self.gn(y.transpose(1, 2).reshape(B * T, H * dv)).view(B, T, H * dv)
        return self.wo(F.silu(self.wg(x)) * y)

    def forward(self, x, mode="parallel", chunk=64):
        q, k, v = self._qkv(x)
        g = self.gamma.double() if x.dtype == torch.float64 else self.gamma
        y = retention_parallel(q, k, v, g) if mode == "parallel" else retention_chunkwise(q, k, v, g, chunk)[0]
        return self._out(x, y)

    def step(self, x_t, state, pos):
        """x_t: [B, 1, d]; state: [B, H, dk, dv] or None."""
        q, k, v = self._qkv(x_t, pos)
        if state is None:
            state = torch.zeros(x_t.size(0), self.h, self.dk, self.dv, dtype=x_t.dtype, device=x_t.device)
        g = self.gamma.double() if x_t.dtype == torch.float64 else self.gamma
        o, state = retention_recurrent_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], state, g)
        return self._out(x_t, o.unsqueeze(2)), state


class RetNetBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.msr = MultiScaleRetention(d, h)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mode="parallel", chunk=64):
        y = self.msr(self.ln1(x), mode, chunk) + x
        return self.ffn(self.ln2(y)) + y

    def step(self, x, state, pos):
        o, state = self.msr.step(self.ln1(x), state, pos)
        y = o + x
        return self.ffn(self.ln2(y)) + y, state


class RetNet(nn.Module):
    def __init__(self, vocab, d=512, n_layers=6, n_heads=4, ff=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([RetNetBlock(d, n_heads, ff) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids, mode="parallel", chunk=64):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, mode, chunk)
        return self.head(self.ln(x))

    @torch.no_grad()
    def step(self, ids_t, states, pos):
        x = self.emb(ids_t)
        new = []
        for b, s in zip(self.blocks, states):
            x, s = b.step(x, s, pos)
            new.append(s)
        return self.head(self.ln(x)), new

    @torch.no_grad()
    def generate(self, prompt, n_new):
        """O(1)-per-token recurrent decoding."""
        states = [None] * len(self.blocks)
        out = prompt
        for t in range(prompt.size(1)):
            logits, states = self.step(prompt[:, t:t + 1], states, t)
        for t in range(prompt.size(1), prompt.size(1) + n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
            logits, states = self.step(nxt, states, t)
        return out
