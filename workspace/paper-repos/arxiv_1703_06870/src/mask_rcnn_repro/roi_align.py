"""RoIAlign (Section 3, Fig. 3): no quantisation of RoI boundaries or bins; features at regularly
sampled points in each bin are computed by bilinear interpolation and averaged (the paper notes max or
average give similar results; 4 samples per bin, i.e. sampling_ratio = 2).

With aligned=True (default) feature cell (i, j) is centred at ((j + 0.5) / scale, (i + 0.5) / scale) in
image space — the half-pixel-corrected variant later adopted by Detectron2; aligned=False reproduces the
original Detectron convention. Pure PyTorch.
"""
from __future__ import annotations

import torch


def bilinear(feat, y, x):
    """feat [C, H, W]; y, x [...] continuous coordinates in feature-cell units (centre of cell i at i).
    Out-of-range samples (beyond one cell outside) are zero, as in the reference implementation."""
    C, H, W = feat.shape
    valid = (y > -1) & (y < H) & (x > -1) & (x < W)
    y, x = y.clamp(0, H - 1), x.clamp(0, W - 1)
    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp(max=H - 1), (x0 + 1).clamp(max=W - 1)
    ly, lx = y - y0, x - x0
    hy, hx = 1 - ly, 1 - lx
    v = (feat[:, y0, x0] * (hy * hx) + feat[:, y0, x1] * (hy * lx) + feat[:, y1, x0] * (ly * hx) + feat[:, y1, x1] * (ly * lx))
    return v * valid


def roi_align(feat, rois, output_size, spatial_scale, sampling_ratio=2, aligned=True):
    """feat [B, C, H, W]; rois [K, 5] (batch_idx, x0, y0, x1, y1) in image coordinates.
    Returns [K, C, out, out]."""
    K = rois.size(0)
    C = feat.size(1)
    out = feat.new_zeros(K, C, output_size, output_size)
    if K == 0:
        return out
    s = sampling_ratio
    g = (torch.arange(s, device=feat.device, dtype=feat.dtype) + 0.5) / s           # sample offsets inside a bin
    b = torch.arange(output_size, device=feat.device, dtype=feat.dtype)
    for k in range(K):
        bi = int(rois[k, 0])
        x0, y0, x1, y1 = (rois[k, 1:] * spatial_scale - (0.5 if aligned else 0.0)).unbind()
        bw, bh = (x1 - x0) / output_size, (y1 - y0) / output_size
        ys = y0 + (b[:, None] + g[None]) * bh                                        # [out, s]
        xs = x0 + (b[:, None] + g[None]) * bw
        Y = ys.reshape(-1)[:, None].expand(-1, output_size * s)
        X = xs.reshape(-1)[None, :].expand(output_size * s, -1)
        v = bilinear(feat[bi], Y, X)                                                 # [C, out*s, out*s]
        out[k] = v.view(C, output_size, s, output_size, s).mean((2, 4))
    return out


def fpn_level(rois, k0=4, canonical=224, kmin=2, kmax=5):
    """Eq. 1 of the FPN paper used by Mask R-CNN: k = floor(k0 + log2(sqrt(wh) / 224)), clamped."""
    w, h = rois[:, 3] - rois[:, 1], rois[:, 4] - rois[:, 2]
    k = torch.floor(k0 + torch.log2(torch.sqrt((w * h).clamp_min(1e-6)) / canonical + 1e-8))
    return k.clamp(kmin, kmax).long()


def multiscale_roi_align(feats, rois, output_size, strides, sampling_ratio=2, canonical=224):
    """feats: list of FPN maps P2..P5; strides: [4, 8, 16, 32]."""
    levels = fpn_level(rois, canonical=canonical) - 2
    out = feats[0].new_zeros(rois.size(0), feats[0].size(1), output_size, output_size)
    for lvl, (f, st) in enumerate(zip(feats, strides)):
        idx = (levels == lvl).nonzero().flatten()
        if len(idx):
            out[idx] = roi_align(f, rois[idx], output_size, 1.0 / st, sampling_ratio)
    return out
