"""NT-Xent: normalized temperature-scaled cross-entropy loss (Section 2.1, Eq. 1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Loss averaged over all 2N anchors; the other 2(N-1) augmented examples are negatives.

    l(i, j) = -log exp(sim(z_i, z_j)/tau) / sum_{k != i} exp(sim(z_i, z_k)/tau), sim = cosine.
    """
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)
