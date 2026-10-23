"""CIFAR-10/100 (paper) or synthetic images (smoke); standard crop+flip augmentation."""
from __future__ import annotations

import torch

from .synth_images import synthetic_images


def load(cfg: dict, train: bool, g: torch.Generator):
    if cfg["name"] == "synthetic":
        return synthetic_images(cfg["n_train"] if train else cfg["n_test"], cfg["image_size"], cfg["n_classes"], g)
    if cfg["name"] in ("cifar10", "cifar100"):  # pragma: no cover
        import numpy as np
        from torchvision.datasets import CIFAR10, CIFAR100
        ds = (CIFAR10 if cfg["name"] == "cifar10" else CIFAR100)(cfg.get("root", "data"), train=train, download=True)
        return torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).float() / 255.0, torch.tensor(ds.targets)
    raise ValueError(cfg["name"])


def augment(x: torch.Tensor, g: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Random crop with 4-pixel padding + horizontal flip (standard CIFAR augmentation)."""
    B, C, H, W = x.shape
    xp = torch.nn.functional.pad(x, (pad,) * 4, mode="reflect")
    out = torch.empty_like(x)
    for i in range(B):
        t, l = torch.randint(0, 2 * pad + 1, (2,), generator=g).tolist()
        out[i] = xp[i, :, t:t + H, l:l + W]
        if torch.rand(1, generator=g).item() < 0.5:
            out[i] = out[i].flip(-1)
    return out
