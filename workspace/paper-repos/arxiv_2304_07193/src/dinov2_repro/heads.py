"""DINO / iBOT projection head (DINO, Caron et al. 2021; used unchanged by DINOv2 with separate, untied
heads for the image-level and patch-level objectives, Section 4): 3-layer MLP (hidden 2048) ->
bottleneck 256 -> L2 normalisation -> weight-normalised linear layer to K prototypes."""
from __future__ import annotations

from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class DINOHead(nn.Module):
    def __init__(self, d_in, n_prototypes=65536, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck))
        self.last = weight_norm(nn.Linear(bottleneck, n_prototypes, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1.0)
        self.last.parametrizations.weight.original0.requires_grad_(False)  # fixed norm g = 1

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))
