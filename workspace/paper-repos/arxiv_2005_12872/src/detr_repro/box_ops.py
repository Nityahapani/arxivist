"""Box utilities: (cx, cy, w, h) <-> (x0, y0, x1, y1) and generalized IoU (Rezatofighi et al., 2019)."""
from __future__ import annotations

import torch


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)


def box_area(b):
    return (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)


def box_iou(a, b):
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    union = box_area(a)[:, None] + box_area(b)[None] - inter
    return inter / union, union


def generalized_box_iou(a, b):
    """Pairwise GIoU [N, M] for xyxy boxes."""
    iou, union = box_iou(a, b)
    lt, rb = torch.min(a[:, None, :2], b[None, :, :2]), torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb - lt).clamp_min(0).prod(-1)
    return iou - (hull - union) / hull
