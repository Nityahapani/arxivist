"""Synthetic touching-cell images (smoke test) and a loader stub for ISBI 2012 EM stacks."""
from __future__ import annotations

import numpy as np


def synthetic_cells(size: int, n_cells: int, rng: np.random.Generator, radius=(6, 12)):
    """Random overlapping disks (cells) with thin background gaps; returns image, binary labels, instances."""
    yy, xx = np.mgrid[:size, :size]
    inst = np.zeros((size, size), dtype=np.int64)
    for k in range(1, n_cells + 1):
        r = rng.uniform(*radius)
        cy, cx = rng.uniform(r, size - r, 2)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        inst[disk & (inst == 0)] = k
    # carve 1-pixel gaps between touching cells so instances are separable
    from scipy import ndimage
    edges = ndimage.morphological_gradient(inst, size=3) > 0
    inst[edges & (inst > 0)] = 0
    img = (inst > 0).astype(np.float32) * 0.6 + rng.normal(0, 0.15, (size, size)).astype(np.float32)
    return img[None], (inst > 0).astype(np.int64), inst


def load_isbi2012(root: str):  # pragma: no cover - dataset must be downloaded manually
    """ISBI 2012 EM segmentation challenge: 30 training slices (512x512) in multi-page TIFFs."""
    import tifffile

    return tifffile.imread(f"{root}/train-volume.tif"), tifffile.imread(f"{root}/train-labels.tif")
