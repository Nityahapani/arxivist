"""Set prediction loss (Section 3.1, Eq. 2 and Appendix A.2).

L_Hungarian = sum_i [ -log p_hat_{sigma(i)}(c_i) + 1{c_i != ∅} L_box ],
L_box = lambda_L1 ||b - b_hat||_1 + lambda_iou L_GIoU, with lambda_L1 = 5, lambda_iou = 2.
The no-object class is down-weighted by 0.1 in the CE. Box losses are normalised by the number of
objects in the batch. Auxiliary losses are added after each decoder layer (Section 3.2).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .box_ops import cxcywh_to_xyxy, generalized_box_iou
from .matcher import hungarian_match


class SetCriterion(torch.nn.Module):
    def __init__(self, n_classes, eos_coef=0.1, w_ce=1.0, w_l1=5.0, w_giou=2.0):
        super().__init__()
        self.n_classes, self.w = n_classes, dict(ce=w_ce, l1=w_l1, giou=w_giou)
        weight = torch.ones(n_classes + 1)
        weight[-1] = eos_coef
        self.register_buffer("empty_weight", weight)

    def single(self, logits, boxes, targets):
        idx = hungarian_match(logits, boxes, targets, self.w["ce"], self.w["l1"], self.w["giou"])
        tgt_cls = torch.full(logits.shape[:2], self.n_classes, dtype=torch.long, device=logits.device)
        src_b, tgt_b = [], []
        for b, ((i, j), t) in enumerate(zip(idx, targets)):
            tgt_cls[b, i] = t["labels"][j]
            src_b.append(boxes[b, i]); tgt_b.append(t["boxes"][j])
        loss_ce = F.cross_entropy(logits.transpose(1, 2), tgt_cls, self.empty_weight)
        src_b, tgt_b = torch.cat(src_b), torch.cat(tgt_b)
        n = max(1, len(tgt_b))
        loss_l1 = F.l1_loss(src_b, tgt_b, reduction="sum") / n
        loss_giou = (1 - torch.diag(generalized_box_iou(cxcywh_to_xyxy(src_b), cxcywh_to_xyxy(tgt_b)))).sum() / n if len(tgt_b) else src_b.sum() * 0
        return {"ce": loss_ce, "l1": loss_l1, "giou": loss_giou}

    def forward(self, outputs, targets):
        """outputs: dict with 'logits' [L, B, N, C+1] and 'boxes' [L, B, N, 4] for all decoder layers."""
        total, last = 0.0, None
        for logits, boxes in zip(outputs["logits"], outputs["boxes"]):
            parts = self.single(logits, boxes, targets)
            total = total + sum(self.w[k] * v for k, v in parts.items())
            last = parts
        return total, {k: v.item() for k, v in last.items()}
