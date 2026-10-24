"""Datasets: synthetic images (smoke) and CIFAR-10 / ImageNet via torchvision (paper scale)."""
from __future__ import annotations

import torch

from .synth_images import synthetic_images

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_dataset(cfg: dict, train: bool, generator: torch.Generator):
    name = cfg["name"]
    if name == "synthetic":
        n = cfg["n_train"] if train else cfg["n_test"]
        return synthetic_images(n, cfg["image_size"], cfg["n_classes"], generator)
    if name == "cifar10":  # pragma: no cover - requires torchvision + download
        import numpy as np
        from torchvision.datasets import CIFAR10

        ds = CIFAR10(cfg.get("root", "data"), train=train, download=True)
        x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).float() / 255.0
        return x, torch.tensor(ds.targets)
    raise ValueError(f"unknown dataset {name!r} (ImageNet requires a custom streaming loader; see README)")


def normalize(x: torch.Tensor, mean=CIFAR_MEAN, std=CIFAR_STD) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s
