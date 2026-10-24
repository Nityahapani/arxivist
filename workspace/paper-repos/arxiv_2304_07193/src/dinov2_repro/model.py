"""Student / teacher networks: ViT backbone + untied DINO head ([CLS]) and iBOT head (patches).
The teacher is an exponential moving average of the student and receives no gradients."""
from __future__ import annotations

import copy

import torch
from torch import nn

from .heads import DINOHead
from .vit import DinoViT


class DinoV2(nn.Module):
    def __init__(self, backbone: DinoViT, n_prototypes=65536, ibot_prototypes=65536, hidden=2048, bottleneck=256):
        super().__init__()
        self.student = nn.ModuleDict(dict(backbone=backbone, dino_head=DINOHead(backbone.dim, n_prototypes, hidden, bottleneck),
                                          ibot_head=DINOHead(backbone.dim, ibot_prototypes, hidden, bottleneck)))
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self, m: float):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
