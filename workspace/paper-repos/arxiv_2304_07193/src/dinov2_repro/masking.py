"""Random patch masking for iBOT (Section 4): for a fraction of global crops (0.5), mask a ratio sampled
uniformly in [0.1, 0.5] of the patches (block-wise in the reference implementation; random here)."""
from __future__ import annotations

import torch


def sample_masks(batch, n_patches, g, ratio=(0.1, 0.5), p_masked=0.5):
    masks = torch.zeros(batch, n_patches, dtype=torch.bool)
    for b in range(batch):
        if torch.rand(1, generator=g).item() < p_masked:
            r = torch.empty(1).uniform_(*ratio, generator=g).item()
            k = max(1, int(round(r * n_patches)))
            masks[b, torch.randperm(n_patches, generator=g)[:k]] = True
    if not masks.any():
        masks[0, 0] = True
    return masks
