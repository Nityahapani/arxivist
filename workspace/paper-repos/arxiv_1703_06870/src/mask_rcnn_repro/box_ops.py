"""Box utilities (xyxy): IoU, greedy NMS, and the R-CNN box-delta parameterisation."""
from __future__ import annotations

import math

import torch


def box_area(b):
    return (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)


def box_iou(a, b):
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    return inter / (box_area(a)[:, None] + box_area(b)[None] - inter).clamp_min(1e-9)


def nms(boxes, scores, thresh):
    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool)
    iou = box_iou(boxes, boxes)
    for i in order.tolist():
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > thresh
    return torch.tensor(keep, dtype=torch.long)


def batched_nms(boxes, scores, labels, thresh):
    """Per-class NMS via coordinate offsets."""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long)
    off = labels.to(boxes) * (boxes.max() + 1)
    return nms(boxes + off[:, None], scores, thresh)


def encode(ref, gt, w=(1.0, 1.0, 1.0, 1.0)):
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    return torch.stack([w[0] * (gx - rx) / rw, w[1] * (gy - ry) / rh, w[2] * torch.log(gw / rw), w[3] * torch.log(gh / rh)], 1)


def decode(ref, d, w=(1.0, 1.0, 1.0, 1.0), clip=math.log(1000 / 16)):
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    dx, dy = d[:, 0] / w[0], d[:, 1] / w[1]
    dw, dh = (d[:, 2] / w[2]).clamp(max=clip), (d[:, 3] / w[3]).clamp(max=clip)
    cx, cy, pw, ph = rx + dx * rw, ry + dy * rh, rw * torch.exp(dw), rh * torch.exp(dh)
    return torch.stack([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], 1)


def clip_boxes(b, h, w):
    return torch.stack([b[:, 0].clamp(0, w), b[:, 1].clamp(0, h), b[:, 2].clamp(0, w), b[:, 3].clamp(0, h)], 1)
