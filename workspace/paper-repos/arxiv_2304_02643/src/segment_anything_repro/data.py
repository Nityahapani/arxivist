"""Synthetic instance-segmentation images: random coloured ellipses and rectangles on a noisy background
(a CPU-friendly stand-in for SA-1B)."""
from __future__ import annotations

import torch


def synthetic_instances(n, size, max_objects=3, g=None):
    imgs, masks = torch.empty(n, 3, size, size), []
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for i in range(n):
        img = 0.1 * torch.rand(3, size, size, generator=g) + 0.2 * torch.rand(3, 1, 1, generator=g)
        ms = []
        for _ in range(int(torch.randint(1, max_objects + 1, (1,), generator=g))):
            cy, cx = (torch.rand(2, generator=g) * 0.6 + 0.2) * size
            ry, rx = (torch.rand(2, generator=g) * 0.15 + 0.08) * size
            if torch.rand(1, generator=g) < 0.5:
                m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            else:
                m = ((yy - cy).abs() <= ry) & ((xx - cx).abs() <= rx)
            img[:, m] = torch.rand(3, 1, generator=g) * 0.7 + 0.3
            for prev in ms:
                prev &= ~m
            ms.append(m)
        imgs[i] = img
        masks.append([m for m in ms if m.sum() > 4])
    return imgs, masks
