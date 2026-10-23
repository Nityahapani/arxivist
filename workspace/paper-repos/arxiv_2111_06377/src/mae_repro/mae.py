"""Masked autoencoder (He et al., 2021, Section 3).

* Random masking of 75% of patches (uniform, without replacement).
* Encoder: ViT applied to the visible patches only (plus [CLS]).
* Decoder: lightweight Transformer on the full set (encoded visible tokens + shared mask tokens).
* Loss: MSE between reconstructed and target pixels, computed on masked patches only; the target
  is per-patch normalised pixels (Section 3, "normalized pixels").
"""
from __future__ import annotations

import torch
from torch import nn

from .vit import Block, PatchEmbed, sincos_2d

CONFIGS = {
    "vit_tiny_patch4": dict(patch=4, dim=64, depth=2, heads=4),
    "vit_base_patch16": dict(patch=16, dim=768, depth=12, heads=12),
    "vit_large_patch16": dict(patch=16, dim=1024, depth=24, heads=16),
    "vit_huge_patch14": dict(patch=14, dim=1280, depth=32, heads=16),
}


def patchify(imgs: torch.Tensor, p: int) -> torch.Tensor:
    B, C, H, W = imgs.shape
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, p * p * C)


def unpatchify(x: torch.Tensor, p: int, C: int = 3) -> torch.Tensor:
    B, L, _ = x.shape
    h = w = int(L**0.5)
    x = x.reshape(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, C, h * p, w * p)


def random_masking(x: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None):
    """Per-sample random masking by argsort of uniform noise. Returns kept tokens, the binary mask
    (1 = removed) in original order, and the indices that restore the original order."""
    B, L, D = x.shape
    keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, generator=generator).to(x.device) if generator is not None else torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, L, device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_kept, mask, ids_restore


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, in_ch=3, arch="vit_base_patch16", dec_dim=512, dec_depth=8, dec_heads=16,
                 norm_pix_loss=True):
        super().__init__()
        c = CONFIGS[arch]
        self.patch, dim = c["patch"], c["dim"]
        self.in_ch = in_ch
        self.patch_embed = PatchEmbed(img_size, self.patch, in_ch, dim)
        L = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos_embed", torch.from_numpy(sincos_2d(dim, self.patch_embed.grid, True)).float()[None])
        self.blocks = nn.ModuleList([Block(dim, c["heads"]) for _ in range(c["depth"])])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.decoder_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos_embed", torch.from_numpy(sincos_2d(dec_dim, self.patch_embed.grid, True)).float()[None])
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.dec_pred = nn.Linear(dec_dim, self.patch**2 * in_ch)
        self.norm_pix_loss = norm_pix_loss
        self.num_patches = L
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_encoder(self, imgs, mask_ratio, generator=None):
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:]
        x, mask, ids_restore = random_masking(x, mask_ratio, generator)
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], 1)
        for b in self.blocks:
            x = b(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, L = ids_restore.shape
        mask_tokens = self.mask_token.expand(B, L + 1 - x.size(1), -1)
        x_ = torch.cat([x[:, 1:], mask_tokens], 1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.size(2)))
        x = torch.cat([x[:, :1], x_], 1) + self.dec_pos_embed
        for b in self.dec_blocks:
            x = b(x)
        return self.dec_pred(self.dec_norm(x))[:, 1:]

    def loss(self, imgs, pred, mask):
        target = patchify(imgs, self.patch)
        if self.norm_pix_loss:
            mean = target.mean(-1, keepdim=True)
            var = target.var(-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        per_patch = ((pred - target) ** 2).mean(-1)
        return (per_patch * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=0.75, generator=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, generator)
        pred = self.forward_decoder(latent, ids_restore)
        return self.loss(imgs, pred, mask), pred, mask

    @torch.no_grad()
    def features(self, imgs):
        """Encoder features without masking (global average of patch tokens), for probing/fine-tuning."""
        latent, _, _ = self.forward_encoder(imgs, 0.0)
        return latent[:, 1:].mean(1)
