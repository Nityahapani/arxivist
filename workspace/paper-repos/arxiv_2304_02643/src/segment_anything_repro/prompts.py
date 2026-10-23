"""Interactive prompt simulation used in training (Section 3, Appendix A).

The first prompt is a foreground point sampled uniformly from the GT mask, or a box (GT box with noise of
std 10% of the side length, capped at 20 px), with equal probability. Subsequent points are sampled
uniformly from the error region between the previous prediction and GT (positive if a false negative,
negative if a false positive); the previous mask logits are also given as a prompt. The paper uses
11 iterations per mask: 1 initial, 8 iteratively sampled points, and 2 with no new points.
"""
from __future__ import annotations

import torch


def mask_to_box(m):
    ys, xs = torch.nonzero(m, as_tuple=True)
    return torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).float()


def noisy_box(m, g=None, frac=0.1, cap=20.0):
    b = mask_to_box(m)
    wh = torch.stack([b[2] - b[0], b[3] - b[1]]).repeat(2)
    std = (frac * wh).clamp(max=cap)
    return (b + torch.randn(4, generator=g) * std).clamp(0, m.shape[-1] - 1)


def sample_point(region, g=None):
    idx = torch.nonzero(region)
    if len(idx) == 0:
        return None
    y, x = idx[torch.randint(0, len(idx), (1,), generator=g)].unbind(-1)
    return torch.cat([x, y]).float()


def corrective_point(pred, gt, g=None):
    """Sample a point uniformly from the error region; label 1 for false-negative, 0 for false-positive."""
    fn, fp = gt & ~pred, pred & ~gt
    err = fn | fp
    p = sample_point(err, g)
    if p is None:
        return None, None
    x, y = p.long()
    return p, int(fn[y, x].item())
