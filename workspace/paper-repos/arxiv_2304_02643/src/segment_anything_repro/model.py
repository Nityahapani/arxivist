"""SAM = image encoder + prompt encoder + mask decoder. The image embedding is computed once and reused
for any number of prompts (amortised real-time interaction)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class SAM(nn.Module):
    def __init__(self, img_size=1024, patch=16, dim=1280, depth=32, heads=16, window_size=14, global_attn=None,
                 embed_dim=256, dec_depth=2, dec_heads=8, dec_mlp=2048):
        super().__init__()
        hw = (img_size // patch, img_size // patch)
        self.img_size = img_size
        self.image_encoder = ImageEncoderViT(img_size, patch, dim, depth, heads, window_size, global_attn, embed_dim)
        self.prompt_encoder = PromptEncoder(embed_dim, hw, (img_size, img_size))
        self.mask_decoder = MaskDecoder(embed_dim, 3, dec_depth, dec_heads, dec_mlp)

    def predict(self, img_emb, points=None, boxes=None, mask_input=None, multimask=True):
        sparse, dense = self.prompt_encoder(points, boxes, mask_input)
        low, iou = self.mask_decoder(img_emb, self.prompt_encoder.dense_pe(), sparse, dense, multimask)
        return low, iou  # low-res mask logits at 4x the embedding resolution (= img/4)

    def upscale(self, low):
        return F.interpolate(low, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
