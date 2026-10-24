"""Box and mask heads (Section 3, Fig. 4 right, FPN variant).

Box head: RoIAlign 7x7 -> 2 x FC(1024) -> class logits (K + 1) and class-specific box deltas
(weights 10, 10, 5, 5). Mask head: RoIAlign 14x14 -> 4 x conv3x3(256) -> deconv 2x2 stride 2 -> 1x1 conv
to K masks of 28x28. L_mask is the average per-pixel binary cross-entropy on the mask of the GT class
only (decoupled from classification; Section 3). RoIs: 512 per image, 25% positives (IoU >= 0.5).
Mask targets: the GT mask cropped to the RoI and resampled to 28x28 (via RoIAlign on the GT mask).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .box_ops import batched_nms, box_iou, clip_boxes, decode, encode
from .roi_align import multiscale_roi_align, roi_align

BOX_W = (10.0, 10.0, 5.0, 5.0)


def to_rois(boxes_per_image):
    return torch.cat([torch.cat([torch.full((len(b), 1), i, dtype=b.dtype, device=b.device), b], 1) for i, b in enumerate(boxes_per_image)])


def mask_loss(mask_logits, labels, targets):
    """mask_logits [R, K, M, M]; labels [R] (GT classes of positive RoIs); targets [R, M, M]."""
    if len(labels) == 0:
        return mask_logits.sum() * 0
    sel = mask_logits[torch.arange(len(labels)), labels]
    return F.binary_cross_entropy_with_logits(sel, targets)


class RoIHeads(nn.Module):
    def __init__(self, d, n_classes, strides=(4, 8, 16, 32), fc=1024, mask_dim=256, canonical=224,
                 batch_per_image=512, pos_fraction=0.25, score_thresh=0.05, nms_thresh=0.5, detections=100):
        super().__init__()
        self.K, self.strides, self.canonical = n_classes, strides, canonical
        self.box_fc = nn.Sequential(nn.Flatten(), nn.Linear(d * 49, fc), nn.ReLU(), nn.Linear(fc, fc), nn.ReLU())
        self.cls = nn.Linear(fc, n_classes + 1)
        self.reg = nn.Linear(fc, 4 * n_classes)
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(d if i == 0 else mask_dim, mask_dim, 3, padding=1), nn.ReLU()]
        self.mask_convs = nn.Sequential(*layers)
        self.mask_up = nn.ConvTranspose2d(mask_dim, mask_dim, 2, 2)
        self.mask_pred = nn.Conv2d(mask_dim, n_classes, 1)
        self.batch_per_image, self.pos_fraction = batch_per_image, pos_fraction
        self.score_thresh, self.nms_thresh, self.detections = score_thresh, nms_thresh, detections

    def _pool(self, feats, rois, size):
        return multiscale_roi_align(feats[:4], rois, size, self.strides, 2, self.canonical)

    def mask_logits(self, feats, rois):
        return self.mask_pred(F.relu(self.mask_up(self.mask_convs(self._pool(feats, rois, 14)))))

    def sample(self, proposals, targets, g=None):
        out = []
        for p, t in zip(proposals, targets):
            p = torch.cat([p, t["boxes"]])  # add GT boxes as proposals
            iou = box_iou(p, t["boxes"])
            best, idx = iou.max(1)
            pos, neg = (best >= 0.5).nonzero().flatten(), (best < 0.5).nonzero().flatten()
            n_pos = min(len(pos), int(self.batch_per_image * self.pos_fraction))
            pos = pos[torch.randperm(len(pos), generator=g)[:n_pos]]
            neg = neg[torch.randperm(len(neg), generator=g)[: self.batch_per_image - n_pos]]
            keep = torch.cat([pos, neg])
            labels = t["labels"][idx[keep]].clone()
            labels[len(pos):] = self.K  # background
            out.append((p[keep], labels, idx[keep], len(pos)))
        return out

    def forward(self, feats, proposals, image_size, targets=None, g=None):
        if targets is not None:
            samples = self.sample(proposals, targets, g)
            rois = to_rois([s[0] for s in samples])
            labels = torch.cat([s[1] for s in samples])
            x = self.box_fc(self._pool(feats, rois, 7))
            logits, deltas = self.cls(x), self.reg(x).view(-1, self.K, 4)
            loss_cls = F.cross_entropy(logits, labels)
            fg = (labels < self.K).nonzero().flatten()
            gt_boxes = torch.cat([t["boxes"][s[2]] for s, t in zip(samples, targets)])
            reg_t = encode(rois[fg, 1:], gt_boxes[fg], BOX_W)
            loss_box = F.smooth_l1_loss(deltas[fg, labels[fg]], reg_t, beta=1.0, reduction="sum") / max(1, len(labels))
            # mask branch on positive RoIs only
            pos_rois, pos_labels, mask_t = [], [], []
            for b, (s, t) in enumerate(zip(samples, targets)):
                n = s[3]
                if n == 0:
                    continue
                r = torch.cat([torch.full((n, 1), b, dtype=s[0].dtype), s[0][:n]], 1)
                gm = t["masks"][s[2][:n]].float()  # [n, H, W]
                # crop + resample each GT mask with its RoI (mask target = intersection of RoI and GT mask)
                tgt = torch.stack([roi_align(gm[i][None, None], torch.cat([torch.zeros(1, 1), r[i:i + 1, 1:]], 1), 28, 1.0)[0, 0] for i in range(n)])
                pos_rois.append(r); pos_labels.append(s[1][:n]); mask_t.append((tgt >= 0.5).float())
            if pos_rois:
                ml = self.mask_logits(feats, torch.cat(pos_rois))
                loss_mask = mask_loss(ml, torch.cat(pos_labels), torch.cat(mask_t))
            else:
                loss_mask = sum(p.sum() for p in self.mask_pred.parameters()) * 0
            return {"cls": loss_cls, "box": loss_box, "mask": loss_mask}
        return self.inference(feats, proposals, image_size)

    @torch.no_grad()
    def inference(self, feats, proposals, image_size):
        rois = to_rois(proposals)
        x = self.box_fc(self._pool(feats, rois, 7))
        probs, deltas = self.cls(x).softmax(-1), self.reg(x).view(-1, self.K, 4)
        results = []
        for b in range(len(proposals)):
            m = rois[:, 0] == b
            r, p, dl = rois[m, 1:], probs[m, :-1], deltas[m]
            boxes = clip_boxes(decode(r.repeat_interleave(self.K, 0), dl.reshape(-1, 4), BOX_W), *image_size).view(-1, self.K, 4)
            cls = torch.arange(self.K).expand_as(p)
            keep = p > self.score_thresh
            bx, sc, lb = boxes[keep], p[keep], cls[keep]
            k = batched_nms(bx, sc, lb, self.nms_thresh)[: self.detections]
            bx, sc, lb = bx[k], sc[k], lb[k]
            masks = torch.zeros(0, 28, 28)
            if len(bx):  # mask branch applied to the top detections (Section 3.1, inference)
                ml = self.mask_logits(feats, torch.cat([torch.full((len(bx), 1), b, dtype=bx.dtype), bx], 1))
                masks = ml[torch.arange(len(lb)), lb].sigmoid()
            results.append({"boxes": bx, "scores": sc, "labels": lb, "masks": masks})
        return results


def paste_masks(masks, boxes, h, w, thresh=0.5):
    """Resize each 28x28 mask to its box and binarise at 0.5 (Section 3.1)."""
    out = torch.zeros(len(masks), h, w, dtype=torch.bool)
    for i, (m, b) in enumerate(zip(masks, boxes)):
        x0, y0, x1, y1 = b.round().long().tolist()
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        r = F.interpolate(m[None, None], size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0] >= thresh
        ys, xs = slice(max(y0, 0), min(y1, h)), slice(max(x0, 0), min(x1, w))
        out[i, ys, xs] = r[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]
    return out
