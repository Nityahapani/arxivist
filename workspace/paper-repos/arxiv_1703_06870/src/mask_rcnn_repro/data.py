"""Synthetic instance segmentation data: random discs (class 0), squares (class 1) and triangles (class 2)
with per-instance binary masks; COCO needs an external loader (pycocotools)."""
from __future__ import annotations

import torch


def synthetic_instances(batch, size, max_objects, g, n_classes=3):
    yy, xx = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    imgs = torch.rand(batch, 3, size, size, generator=g) * 0.2
    targets = []
    for b in range(batch):
        k = int(torch.randint(1, max_objects + 1, (1,), generator=g))
        masks, labels, boxes = [], [], []
        occupied = torch.zeros(size, size, dtype=torch.bool)
        for _ in range(k):
            c = int(torch.randint(0, n_classes, (1,), generator=g))
            r = float(torch.empty(1).uniform_(size * 0.08, size * 0.2, generator=g))
            cy, cx = torch.empty(2).uniform_(r, size - r, generator=g).tolist()
            if c == 0:
                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            elif c == 1:
                m = ((yy - cy).abs() <= r) & ((xx - cx).abs() <= r)
            else:
                m = (yy >= cy - r) & (yy <= cy + r) & ((xx - cx).abs() <= (yy - (cy - r)) / 2)
            m = m & ~occupied
            if m.sum() < 4:
                continue
            occupied |= m
            imgs[b][:, m] = torch.rand(3, 1, generator=g) * 0.6 + 0.4
            ys, xs = m.nonzero(as_tuple=True)
            boxes.append([xs.min().item(), ys.min().item(), xs.max().item() + 1.0, ys.max().item() + 1.0])
            masks.append(m); labels.append(c)
        if not masks:
            m = torch.zeros(size, size, dtype=torch.bool); m[2:10, 2:10] = True
            masks, labels, boxes = [m], [1], [[2.0, 2.0, 10.0, 10.0]]
            imgs[b][:, m] = 0.9
        targets.append({"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "masks": torch.stack(masks)})
    return imgs, targets
