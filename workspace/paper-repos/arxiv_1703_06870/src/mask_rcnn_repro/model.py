"""Mask R-CNN = backbone (ResNet-FPN) + RPN + RoI heads with a parallel mask branch (Section 3).
Multi-task loss L = L_cls + L_box + L_mask (+ RPN losses)."""
from __future__ import annotations

from torch import nn

from .backbone import MultiScaleResNet, freeze_bn
from .fpn import FPN
from .roi_heads import RoIHeads
from .rpn import RPN


class MaskRCNN(nn.Module):
    def __init__(self, n_classes, arch="resnet50", base_width=64, fpn_dim=256, anchor_sizes=(32, 64, 128, 256, 512),
                 canonical=224, rpn_pre_nms=2000, rpn_post_nms=1000, rois_per_image=512, fc=1024, mask_dim=256):
        super().__init__()
        self.backbone = freeze_bn(MultiScaleResNet(arch, 1, False, base_width))
        self.fpn = FPN(self.backbone.channels, fpn_dim)
        self.rpn = RPN(fpn_dim, anchor_sizes, pre_nms=rpn_pre_nms, post_nms=rpn_post_nms)
        self.roi_heads = RoIHeads(fpn_dim, n_classes, fc=fc, mask_dim=mask_dim, canonical=canonical, batch_per_image=rois_per_image)

    def train(self, mode=True):
        super().train(mode)
        freeze_bn(self.backbone)
        return self

    def forward(self, images, targets=None, g=None):
        size = images.shape[-2:]
        feats = self.fpn(self.backbone(images))
        proposals, rpn_losses = self.rpn(feats, size, targets, g)
        if targets is not None:
            return {**rpn_losses, **self.roi_heads(feats, proposals, size, targets, g)}
        return self.roi_heads(feats, proposals, size)
