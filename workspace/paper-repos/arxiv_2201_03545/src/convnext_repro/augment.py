"""Mixup / CutMix with label smoothing (training recipe, Section 2.1 / Appendix A.1)."""
from __future__ import annotations

import numpy as np
import torch


def one_hot_smooth(y, n, smoothing):
    off = smoothing / n
    return torch.full((y.size(0), n), off).scatter_(1, y.view(-1, 1), 1.0 - smoothing + off)


def mixup_cutmix(x, y, n_classes, mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5, smoothing=0.1, rng=None):
    """Batch-level Mixup or CutMix (chosen with prob. switch_prob), returning soft targets."""
    rng = rng or np.random.default_rng()
    t = one_hot_smooth(y, n_classes, smoothing)
    perm = torch.arange(x.size(0) - 1, -1, -1)  # flip batch, as in timm
    if rng.random() < switch_prob and cutmix_alpha > 0:
        lam = rng.beta(cutmix_alpha, cutmix_alpha)
        H, W = x.shape[-2:]
        rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = rng.integers(H), rng.integers(W)
        y0, y1 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
        x0, x1 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    else:
        lam = rng.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * t + (1 - lam) * t[perm]


def soft_ce(logits, target):
    return torch.sum(-target * torch.log_softmax(logits, -1), -1).mean()
