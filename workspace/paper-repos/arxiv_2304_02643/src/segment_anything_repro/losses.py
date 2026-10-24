"""Training losses (Section 3, Appendix A): a linear combination of focal and dice loss at 20:1, plus an
MSE loss between the IoU prediction and the IoU of the predicted mask with the ground truth. With
multiple output masks, only the mask with the lowest loss is back-propagated (ambiguity-aware)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - p_t) ** gamma
    loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.flatten(-2).mean(-1)


def dice_loss(logits, target, eps=1.0):
    p = logits.sigmoid().flatten(-2)
    t = target.flatten(-2)
    return 1 - (2 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)


def mask_iou(logits, target):
    pred = (logits > 0).float().flatten(-2)
    t = target.flatten(-2)
    inter = (pred * t).sum(-1)
    union = pred.sum(-1) + t.sum(-1) - inter
    return inter / union.clamp_min(1.0)


def sam_loss(masks, iou_pred, target, focal_w=20.0, dice_w=1.0, iou_w=1.0):
    """masks [B, K, H, W] logits, iou_pred [B, K], target [B, H, W] in {0,1}. Returns (loss, best index)."""
    t = target.unsqueeze(1).expand_as(masks)
    per_mask = focal_w * focal_loss(masks, t) + dice_w * dice_loss(masks, t)
    best = per_mask.argmin(1)
    ar = torch.arange(masks.shape[0], device=masks.device)
    iou_t = mask_iou(masks.detach(), t)
    iou_l = F.mse_loss(iou_pred, iou_t)
    return per_mask[ar, best].mean() + iou_w * iou_l, best
