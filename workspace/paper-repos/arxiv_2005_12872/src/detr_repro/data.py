"""Synthetic detection data: random coloured axis-aligned rectangles on a noisy canvas. Each object's
class is its colour; boxes are normalised (cx, cy, w, h). COCO requires an external loader."""
from __future__ import annotations

import torch

COLORS = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0], [1.0, 1.0, 0.1], [1.0, 0.1, 1.0], [0.1, 1.0, 1.0]])


def synthetic_detection(batch, size, n_classes, max_objects, g):
    imgs = torch.rand(batch, 3, size, size, generator=g) * 0.2
    targets = []
    for b in range(batch):
        k = int(torch.randint(1, max_objects + 1, (1,), generator=g))
        labels, boxes = [], []
        for _ in range(k):
            c = int(torch.randint(0, n_classes, (1,), generator=g))
            w, h = (torch.rand(2, generator=g) * 0.35 + 0.1).tolist()
            x0, y0 = (torch.rand(1, generator=g).item() * (1 - w), torch.rand(1, generator=g).item() * (1 - h))
            X0, Y0, X1, Y1 = int(x0 * size), int(y0 * size), max(int((x0 + w) * size), int(x0 * size) + 1), max(int((y0 + h) * size), int(y0 * size) + 1)
            imgs[b, :, Y0:Y1, X0:X1] = COLORS[c % len(COLORS)].view(3, 1, 1)
            labels.append(c)
            boxes.append([(X0 + X1) / 2 / size, (Y0 + Y1) / 2 / size, (X1 - X0) / size, (Y1 - Y0) / size])
        targets.append({"labels": torch.tensor(labels), "boxes": torch.tensor(boxes)})
    return imgs, targets
