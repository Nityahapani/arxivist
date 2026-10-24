"""Region Proposal Network with FPN (Ren et al., 2015; Lin et al., 2017), as used by Mask R-CNN.

Anchors: one size per pyramid level (P2..P6) and 3 aspect ratios. Training: anchors with IoU > 0.7 (or
the best anchor for each GT) are positive, IoU < 0.3 negative; 256 anchors per image with up to 50%
positives; BCE objectiveness + smooth-L1 box loss. Proposals: top-k per level pre-NMS, NMS 0.7, top-N.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .box_ops import box_iou, clip_boxes, decode, encode, nms


def make_anchors(sizes, ratios, strides, shapes, device=None):
    out = []
    for size, st, (h, w) in zip(sizes, strides, shapes):
        r = torch.tensor(ratios, device=device, dtype=torch.float32)
        ws, hs = size / torch.sqrt(r), size * torch.sqrt(r)
        base = torch.stack([-ws, -hs, ws, hs], 1) / 2                                   # [A, 4]
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        ctr = torch.stack([xs, ys, xs, ys], -1).reshape(-1, 1, 4).float() * st + st / 2
        out.append((ctr + base[None]).reshape(-1, 4))
    return out


class RPNHead(nn.Module):
    def __init__(self, d, n_anchors):
        super().__init__()
        self.conv = nn.Conv2d(d, d, 3, padding=1)
        self.cls = nn.Conv2d(d, n_anchors, 1)
        self.reg = nn.Conv2d(d, 4 * n_anchors, 1)

    def forward(self, feats):
        logits, deltas = [], []
        for f in feats:
            h = F.relu(self.conv(f))
            B, _, H, W = h.shape
            logits.append(self.cls(h).permute(0, 2, 3, 1).reshape(B, -1))
            deltas.append(self.reg(h).view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 4))
        return logits, deltas


class RPN(nn.Module):
    def __init__(self, d=256, sizes=(32, 64, 128, 256, 512), ratios=(0.5, 1.0, 2.0), strides=(4, 8, 16, 32, 64),
                 pre_nms=2000, post_nms=1000, nms_thresh=0.7, batch_per_image=256, pos_fraction=0.5):
        super().__init__()
        self.sizes, self.ratios, self.strides = sizes, ratios, strides
        self.head = RPNHead(d, len(ratios))
        self.pre_nms, self.post_nms, self.nms_thresh = pre_nms, post_nms, nms_thresh
        self.batch_per_image, self.pos_fraction = batch_per_image, pos_fraction

    def label_anchors(self, anchors, gt):
        labels = torch.full((len(anchors),), -1, dtype=torch.long)
        if len(gt) == 0:
            labels[:] = 0
            return labels, torch.zeros_like(anchors)
        iou = box_iou(anchors, gt)
        best, idx = iou.max(1)
        labels[best < 0.3] = 0
        labels[best >= 0.7] = 1
        labels[iou.argmax(0)] = 1  # best anchor for each GT
        return labels, gt[idx]

    def sample(self, labels, g=None):
        pos, neg = (labels == 1).nonzero().flatten(), (labels == 0).nonzero().flatten()
        n_pos = min(len(pos), int(self.batch_per_image * self.pos_fraction))
        n_neg = min(len(neg), self.batch_per_image - n_pos)
        pos = pos[torch.randperm(len(pos), generator=g)[:n_pos]]
        neg = neg[torch.randperm(len(neg), generator=g)[:n_neg]]
        return pos, neg

    def forward(self, feats, image_size, targets=None, g=None):
        logits, deltas = self.head(feats)
        anchors = make_anchors(self.sizes, self.ratios, self.strides, [f.shape[-2:] for f in feats], feats[0].device)
        B = feats[0].size(0)
        proposals = []
        for b in range(B):
            boxes, scores = [], []
            for lg, dl, an in zip(logits, deltas, anchors):
                k = min(self.pre_nms, len(an))
                s, i = lg[b].detach().topk(k)
                boxes.append(clip_boxes(decode(an[i], dl[b, i].detach()), *image_size)); scores.append(s)
            boxes, scores = torch.cat(boxes), torch.cat(scores)
            keep_wh = ((boxes[:, 2] - boxes[:, 0]) > 1e-3) & ((boxes[:, 3] - boxes[:, 1]) > 1e-3)
            boxes, scores = boxes[keep_wh], scores[keep_wh]
            keep = nms(boxes, scores, self.nms_thresh)[: self.post_nms]
            proposals.append(boxes[keep])
        losses = {}
        if targets is not None:
            all_anchors = torch.cat(anchors)
            lg_all, dl_all = torch.cat(logits, 1), torch.cat(deltas, 1)
            cls_l, reg_l = 0.0, 0.0
            for b, t in enumerate(targets):
                labels, matched = self.label_anchors(all_anchors, t["boxes"])
                pos, neg = self.sample(labels, g)
                idx = torch.cat([pos, neg])
                cls_l = cls_l + F.binary_cross_entropy_with_logits(lg_all[b, idx], (labels[idx] == 1).float())
                if len(pos):
                    reg_l = reg_l + F.smooth_l1_loss(dl_all[b, pos], encode(all_anchors[pos], matched[pos]), beta=1 / 9, reduction="sum") / max(1, len(idx))
            losses = {"rpn_cls": cls_l / B, "rpn_box": reg_l / B}
        return proposals, losses
