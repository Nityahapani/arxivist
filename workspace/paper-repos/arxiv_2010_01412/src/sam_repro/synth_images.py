"""Synthetic labelled images for CPU smoke tests (not a paper dataset).

Each image is low-amplitude noise plus a square whose colour encodes the class, placed at a
random location, so labels are learnable but augmentation-invariant.
"""
from __future__ import annotations

import torch


def synthetic_images(n: int, size: int = 32, n_classes: int = 10, generator: torch.Generator | None = None):
    g = generator
    palette = torch.rand(n_classes, 3, generator=torch.Generator().manual_seed(1234))
    y = torch.randint(0, n_classes, (n,), generator=g)
    x = 0.1 * torch.rand(n, 3, size, size, generator=g)
    s = max(4, size // 3)
    for i in range(n):
        t = torch.randint(0, size - s + 1, (1,), generator=g).item()
        l = torch.randint(0, size - s + 1, (1,), generator=g).item()
        x[i, :, t:t + s, l:l + s] = palette[y[i]].view(3, 1, 1)
    return x.clamp(0, 1), y
