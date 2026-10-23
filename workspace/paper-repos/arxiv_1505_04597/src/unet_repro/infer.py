"""Overlap-tile inference (Section 2, Fig. 2): mirror-pad the image and predict tiles whose valid
outputs cover the whole image seamlessly."""
from __future__ import annotations

import numpy as np
import torch

from .model import output_size


@torch.no_grad()
def overlap_tile_predict(model, image: np.ndarray, tile_in: int, depth: int = 4) -> np.ndarray:
    """image: [C, H, W]. Returns class scores [n_classes, H, W]."""
    model.eval()
    tile_out = output_size(tile_in, depth)
    margin = (tile_in - tile_out) // 2
    C, H, W = image.shape
    ny, nx = -(-H // tile_out), -(-W // tile_out)
    padded = np.pad(image, ((0, 0), (margin, margin + ny * tile_out - H), (margin, margin + nx * tile_out - W)), mode="reflect")
    out = None
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * tile_out, ix * tile_out
            tile = torch.from_numpy(padded[None, :, y0:y0 + tile_in, x0:x0 + tile_in].copy()).float()
            pred = model(tile)[0].numpy()
            if out is None:
                out = np.zeros((pred.shape[0], ny * tile_out, nx * tile_out), dtype=np.float32)
            out[:, y0:y0 + tile_out, x0:x0 + tile_out] = pred
    return out[:, :H, :W]
