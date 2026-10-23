"""Pixel-wise loss weight map (Section 3, Eq. 2):

w(x) = w_c(x) + w_0 * exp(-(d_1(x) + d_2(x))^2 / (2 sigma^2)),  w_0 = 10, sigma ≈ 5 pixels,
where d_1, d_2 are distances to the border of the nearest and second-nearest cell and w_c balances
class frequencies.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def class_balance_weights(labels: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(classes, counts / labels.size))
    wc = np.zeros(labels.shape, dtype=np.float64)
    for c, f in freq.items():
        wc[labels == c] = 1.0 / (len(classes) * f)
    return wc


def unet_weight_map(instances: np.ndarray, w0: float = 10.0, sigma: float = 5.0) -> np.ndarray:
    """instances: int array, 0 = background, k > 0 = cell id. Returns float weight map."""
    fg = instances > 0
    wc = class_balance_weights(fg.astype(np.int64))
    ids = [i for i in np.unique(instances) if i != 0]
    if len(ids) < 2:
        return wc
    dists = np.stack([ndimage.distance_transform_edt(instances != i) for i in ids], 0)
    dists.sort(axis=0)
    d1, d2 = dists[0], dists[1]
    border = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))
    return wc + border * (~fg)  # separation term applies to background pixels between cells
