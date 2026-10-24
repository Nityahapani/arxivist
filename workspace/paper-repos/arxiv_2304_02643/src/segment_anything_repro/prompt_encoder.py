"""Prompt encoder (Section 3, Appendix A).

Sparse prompts: points and boxes are represented by random-Fourier positional encodings (Tancik et al.)
summed with learned embeddings for each prompt type (positive point, negative point, box top-left,
box bottom-right); a learned "not-a-point" embedding pads point sets when no box is given.
Dense prompts: masks are input at 4x lower resolution than the image and downscaled a further 4x by two
2x2 stride-2 convs (channels 4, 16) with LayerNorm + GELU, then a 1x1 conv to 256; if no mask is given a
learned "no mask" embedding is added at every location.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .image_encoder import LayerNorm2d


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=128, scale=1.0):
        super().__init__()
        self.register_buffer("gaussian", scale * torch.randn(2, num_pos_feats))

    def _pe(self, coords):  # coords in [0, 1]
        c = (2 * coords - 1) @ self.gaussian * 2 * math.pi
        return torch.cat([c.sin(), c.cos()], -1)

    def grid(self, h, w):
        ys = (torch.arange(h, dtype=torch.float32, device=self.gaussian.device) + 0.5) / h
        xs = (torch.arange(w, dtype=torch.float32, device=self.gaussian.device) + 0.5) / w
        g = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)  # [h, w, 2] as (x, y)
        return self._pe(g).permute(2, 0, 1)  # [C, h, w]

    def forward_coords(self, coords, img_size):
        c = coords.clone().float()
        c[..., 0] /= img_size[1]
        c[..., 1] /= img_size[0]
        return self._pe(c)


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, embed_hw=(64, 64), img_size=(1024, 1024), mask_chans=16):
        super().__init__()
        self.embed_dim, self.embed_hw, self.img_size = embed_dim, embed_hw, img_size
        self.pe = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(4)])  # neg, pos, box TL, box BR
        self.not_a_point = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * embed_hw[0], 4 * embed_hw[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_chans // 4, 2, 2), LayerNorm2d(mask_chans // 4), nn.GELU(),
            nn.Conv2d(mask_chans // 4, mask_chans, 2, 2), LayerNorm2d(mask_chans), nn.GELU(),
            nn.Conv2d(mask_chans, embed_dim, 1))
        self.no_mask = nn.Embedding(1, embed_dim)

    def dense_pe(self):
        return self.pe.grid(*self.embed_hw).unsqueeze(0)

    def _embed_points(self, pts, labels, pad):
        pts = pts + 0.5  # shift to pixel centre
        if pad:
            pts = torch.cat([pts, torch.zeros_like(pts[:, :1])], 1)
            labels = torch.cat([labels, -torch.ones_like(labels[:, :1])], 1)
        e = self.pe.forward_coords(pts, self.img_size)
        e = torch.where((labels == -1)[..., None], self.not_a_point.weight.expand_as(e), e)
        e = e + torch.where((labels == 0)[..., None], self.point_embeddings[0].weight, torch.zeros_like(e))
        e = e + torch.where((labels == 1)[..., None], self.point_embeddings[1].weight, torch.zeros_like(e))
        return e

    def _embed_boxes(self, boxes):
        e = self.pe.forward_coords(boxes.reshape(-1, 2, 2) + 0.5, self.img_size)
        return torch.stack([e[:, 0] + self.point_embeddings[2].weight[0], e[:, 1] + self.point_embeddings[3].weight[0]], 1)

    def forward(self, points=None, boxes=None, masks=None):
        """points: (coords [B, N, 2] in (x, y) pixels, labels [B, N] with 1=pos 0=neg); boxes [B, 4] xyxy;
        masks [B, 1, 4h, 4w] logits. Returns sparse [B, T, C] and dense [B, C, h, w] embeddings."""
        B = (points[0] if points is not None else boxes if boxes is not None else masks).shape[0]
        sparse = torch.empty(B, 0, self.embed_dim, device=self.no_mask.weight.device)
        if points is not None:
            sparse = torch.cat([sparse, self._embed_points(points[0], points[1], pad=boxes is None)], 1)
        if boxes is not None:
            sparse = torch.cat([sparse, self._embed_boxes(boxes)], 1)
        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            dense = self.no_mask.weight.view(1, -1, 1, 1).expand(B, -1, *self.embed_hw)
        return sparse, dense
