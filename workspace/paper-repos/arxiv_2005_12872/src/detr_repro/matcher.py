"""Hungarian matching (Section 3.1, Eq. 1): optimal bipartite assignment between N predictions and the
ground-truth objects with cost  -p_hat(c_i) + lambda_L1 ||b_i - b_hat||_1 + lambda_giou (-GIoU)."""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import cxcywh_to_xyxy, generalized_box_iou


@torch.no_grad()
def hungarian_match(logits, boxes, targets, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
    """logits [B, N, C+1], boxes [B, N, 4] (cxcywh in [0,1]); targets: list of dicts with 'labels', 'boxes'.
    Returns a list of (pred_idx, tgt_idx) LongTensor pairs."""
    out = []
    prob = logits.softmax(-1)
    for b, t in enumerate(targets):
        if len(t["labels"]) == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        c = -prob[b][:, t["labels"]] * cost_class
        c = c + cost_bbox * torch.cdist(boxes[b], t["boxes"], p=1)
        c = c - cost_giou * generalized_box_iou(cxcywh_to_xyxy(boxes[b]), cxcywh_to_xyxy(t["boxes"]))
        i, j = linear_sum_assignment(c.cpu().numpy())
        out.append((torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)))
    return out
