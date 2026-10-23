"""Mamba-2 block and language model (Section 7, Fig. 6).

Parallel projections: in_proj(u) -> [z, x, B, C, dt]; short causal depthwise conv on (x, B, C);
dt = softplus(dt + dt_bias); per-head scalar A = -exp(A_log); SSD with log_a = dt * A and input dt * x;
skip D; gated RMSNorm y * silu(z); out_proj. Multi-input SSM: B, C shared across heads (ngroups = 1).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .ssd import ssd_chunked, ssd_recurrent


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x, z):
        x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64, head_dim=64, expand=2, d_conv=4, chunk=64, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        self.d_inner = expand * d_model
        self.h, self.p, self.n, self.chunk = self.d_inner // head_dim, head_dim, d_state, chunk
        self.conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.h, bias=False)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, groups=self.conv_dim, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.h).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.h))
        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _split(self, zxbcdt):
        return torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.h], -1)

    def _ssm_inputs(self, xBC, dt):
        x, B, C = torch.split(xBC, [self.d_inner, self.n, self.n], -1)
        dt = F.softplus(dt + self.dt_bias)                                   # [b, l, h]
        A = -torch.exp(self.A_log)
        sh = x.shape[:-1]
        x = x.reshape(*sh, self.h, self.p)
        Bh = B.unsqueeze(-2).expand(*sh, self.h, self.n)
        Ch = C.unsqueeze(-2).expand(*sh, self.h, self.n)
        return x, dt, A, Bh, Ch

    def forward(self, u, mode="chunked"):
        b, l, _ = u.shape
        z, xBC, dt = self._split(self.in_proj(u))
        xBC = F.silu(self.conv(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, dt, A, B, C = self._ssm_inputs(xBC, dt)
        if mode == "chunked":
            pad = (-l) % self.chunk
            args = [F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x * dt[..., None], B, C)]
            y, _ = ssd_chunked(args[0], F.pad(dt * A, (0, 0, 0, pad)), args[1], args[2], self.chunk)
            y = y[:, :l]
        else:
            y, _ = ssd_recurrent(x * dt[..., None], dt * A, B, C)
        y = y + x * self.D[:, None]
        return self.out_proj(self.norm(y.reshape(b, l, self.d_inner), z))

    @torch.no_grad()
    def step(self, u_t, cache):
        """u_t: [b, 1, d]; cache = (conv_state [b, conv_dim, d_conv], ssm_state [b, h, p, n]) or None."""
        b = u_t.size(0)
        z, xBC, dt = self._split(self.in_proj(u_t))
        k = self.conv.kernel_size[0]
        if cache is None:
            cache = (u_t.new_zeros(b, self.conv_dim, k), u_t.new_zeros(b, self.h, self.p, self.n))
        conv_state, ssm_state = cache
        conv_state = torch.cat([conv_state[..., 1:], xBC.transpose(1, 2)], -1)
        xBC = F.silu((conv_state * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias).unsqueeze(1)
        x, dt, A, B, C = self._ssm_inputs(xBC, dt)
        y, ssm_state = ssd_recurrent(x * dt[..., None], dt * A, B, C, ssm_state)
        y = y + x * self.D[:, None]
        return self.out_proj(self.norm(y.reshape(b, 1, self.d_inner), z)), (conv_state, ssm_state)


class Mamba2LM(nn.Module):
    def __init__(self, vocab, d_model=256, n_layers=4, d_state=64, head_dim=64, expand=2, chunk=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([Mamba2Block(d_model, d_state, head_dim, expand, chunk=chunk) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids, mode="chunked"):
        x = self.emb(ids)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x), mode)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def step(self, ids_t, caches):
        x = self.emb(ids_t)
        new = []
        for norm, layer, c in zip(self.norms, self.layers, caches):
            o, c = layer.step(norm(x), c)
            x = x + o
            new.append(c)
        return self.head(self.norm_f(x)), new
