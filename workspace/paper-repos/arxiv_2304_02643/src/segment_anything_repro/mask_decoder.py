"""Lightweight mask decoder (Section 3, Appendix A, Fig. 14).

Output tokens (1 IoU token + 4 mask tokens) are concatenated with the prompt tokens. A two-way
transformer (2 layers, dim 256, 8 heads, MLP 2048) runs per layer: (1) token self-attention,
(2) token-to-image cross-attention, (3) a point-wise MLP on tokens, (4) image-to-token cross-attention.
Positional encodings are re-added to the image embedding and the original prompt tokens re-added to
tokens at every attention layer. Cross-attention uses an internal dim of 128 (downsample 2). A final
token-to-image attention follows. The image embedding is upscaled 4x by two transposed convs; each
mask token passes through a 3-layer MLP (hypernetwork) whose output is dotted with the upscaled
embedding to give a mask. The IoU token predicts each mask's IoU with a 3-layer MLP.
Multimask output returns masks from tokens 1..3; single-mask output uses token 0.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .image_encoder import LayerNorm2d


class MLP(nn.Module):
    def __init__(self, din, dh, dout, n):
        super().__init__()
        dims = [din] + [dh] * (n - 1) + [dout]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x) if i == len(self.layers) - 1 else F.relu(l(x))
        return x


class Attn(nn.Module):
    def __init__(self, dim, heads, downsample=1):
        super().__init__()
        inner = dim // downsample
        self.h = heads
        self.q, self.k, self.v, self.o = nn.Linear(dim, inner), nn.Linear(dim, inner), nn.Linear(dim, inner), nn.Linear(inner, dim)

    def forward(self, q, k, v):
        B = q.shape[0]
        sp = lambda x: x.view(B, x.shape[1], self.h, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(self.q(q)), sp(self.k(k)), sp(self.v(v)))
        return self.o(o.transpose(1, 2).reshape(B, q.shape[1], -1))


class TwoWayBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, skip_first_pe=False):
        super().__init__()
        self.self_attn, self.n1 = Attn(dim, heads), nn.LayerNorm(dim)
        self.t2i, self.n2 = Attn(dim, heads, 2), nn.LayerNorm(dim)
        self.mlp, self.n3 = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, dim)), nn.LayerNorm(dim)
        self.i2t, self.n4 = Attn(dim, heads, 2), nn.LayerNorm(dim)
        self.skip_first_pe = skip_first_pe

    def forward(self, q, k, q_pe, k_pe):
        if self.skip_first_pe:
            q = self.self_attn(q, q, q)
        else:
            qq = q + q_pe
            q = q + self.self_attn(qq, qq, q)
        q = self.n1(q)
        q = self.n2(q + self.t2i(q + q_pe, k + k_pe, k))
        q = self.n3(q + self.mlp(q))
        k = self.n4(k + self.i2t(k + k_pe, q + q_pe, q))
        return q, k


class TwoWayTransformer(nn.Module):
    def __init__(self, depth=2, dim=256, heads=8, mlp_dim=2048):
        super().__init__()
        self.layers = nn.ModuleList(TwoWayBlock(dim, heads, mlp_dim, skip_first_pe=(i == 0)) for i in range(depth))
        self.final, self.nf = Attn(dim, heads, 2), nn.LayerNorm(dim)

    def forward(self, img, img_pe, tokens):
        k, k_pe = img.flatten(2).transpose(1, 2), img_pe.flatten(2).transpose(1, 2)
        q = tokens
        for layer in self.layers:
            q, k = layer(q, k, tokens, k_pe)
        q = self.nf(q + self.final(q + tokens, k + k_pe, k))
        return q, k


class MaskDecoder(nn.Module):
    def __init__(self, dim=256, num_multimask=3, depth=2, heads=8, mlp_dim=2048, iou_hidden=256):
        super().__init__()
        self.n_mask_tokens = num_multimask + 1
        self.iou_token, self.mask_tokens = nn.Embedding(1, dim), nn.Embedding(self.n_mask_tokens, dim)
        self.transformer = TwoWayTransformer(depth, dim, heads, mlp_dim)
        self.upscale = nn.Sequential(nn.ConvTranspose2d(dim, dim // 4, 2, 2), LayerNorm2d(dim // 4), nn.GELU(),
                                     nn.ConvTranspose2d(dim // 4, dim // 8, 2, 2), nn.GELU())
        self.hyper = nn.ModuleList(MLP(dim, dim, dim // 8, 3) for _ in range(self.n_mask_tokens))
        self.iou_head = MLP(dim, iou_hidden, self.n_mask_tokens, 3)

    def forward(self, img_emb, img_pe, sparse, dense, multimask=True):
        B = sparse.shape[0]
        out_tok = torch.cat([self.iou_token.weight, self.mask_tokens.weight]).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([out_tok, sparse], 1)
        src = img_emb.expand(B, -1, -1, -1) + dense
        pe = img_pe.expand(B, -1, -1, -1)
        h, w = src.shape[-2:]
        hs, src = self.transformer(src, pe, tokens)
        iou_out, mask_out = hs[:, 0], hs[:, 1:1 + self.n_mask_tokens]
        up = self.upscale(src.transpose(1, 2).reshape(B, -1, h, w))
        hyper = torch.stack([m(mask_out[:, i]) for i, m in enumerate(self.hyper)], 1)
        masks = (hyper @ up.flatten(2)).view(B, -1, *up.shape[-2:])
        iou = self.iou_head(iou_out)
        sl = slice(1, None) if multimask else slice(0, 1)
        return masks[:, sl], iou[:, sl]
