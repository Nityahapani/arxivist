"""BYOL objective (Section 3.1, Eq. 2): normalised MSE between the online prediction and the target
projection, L = ||q̄ - z̄'||² = 2 - 2 cos(q, z'), symmetrised over the two views (Eq. 3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p, z = F.normalize(p, dim=-1), F.normalize(z.detach(), dim=-1)
    return 2 - 2 * (p * z).sum(-1)


def byol_loss(p1, p2, z1, z2):
    """p_i: online predictions of view i; z_i: target projections of view i. Returns batch mean of L + L~."""
    return (regression_loss(p1, z2) + regression_loss(p2, z1)).mean()
