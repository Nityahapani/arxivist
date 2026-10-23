"""Elastic deformation (Section 3.1): random displacement vectors on a coarse 3x3 grid sampled from a
Gaussian with 10 px standard deviation, interpolated with bicubic interpolation to per-pixel shifts."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def elastic_deform(image: np.ndarray, labels: np.ndarray | None = None, grid: int = 3, sigma: float = 10.0,
                   rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    H, W = image.shape[-2:]
    coarse = rng.normal(0.0, sigma, size=(2, grid, grid))
    dy = ndimage.zoom(coarse[0], (H / grid, W / grid), order=3)[:H, :W]
    dx = ndimage.zoom(coarse[1], (H / grid, W / grid), order=3)[:H, :W]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = [yy + dy, xx + dx]
    def warp(a, order):
        if a.ndim == 2:
            return ndimage.map_coordinates(a, coords, order=order, mode="reflect")
        return np.stack([ndimage.map_coordinates(c, coords, order=order, mode="reflect") for c in a])
    out = warp(image, 3)
    return (out, warp(labels, 0)) if labels is not None else out
