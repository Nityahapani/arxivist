"""First-stage autoencoders (Section 3.1, Appendix G).

Encoder E maps x [B, 3, H, W] to z [B, c, H/f, W/f]; decoder D reconstructs. Two regularisations:
  * KL-reg: a slight KL penalty towards a standard normal on a diagonal-Gaussian posterior (weight ~1e-6);
  * VQ-reg: a vector-quantisation layer absorbed into the decoder (VQGAN-style, commitment 0.25).
Training combines a reconstruction loss, a perceptual loss (LPIPS; optional) and a patch-based
adversarial loss (PatchGAN, hinge) that starts after a warmup.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def Norm(c):
    return nn.GroupNorm(min(32, c // 4) if c >= 8 else 1, c, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = Norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = Norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class Encoder(nn.Module):
    def __init__(self, ch=128, mult=(1, 2, 4, 4), z_ch=4, double_z=True, n_res=2):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1)]
        c = ch
        for i, m in enumerate(mult):
            for _ in range(n_res):
                layers.append(ResBlock(c, ch * m)); c = ch * m
            if i < len(mult) - 1:
                layers.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        layers += [ResBlock(c, c), Norm(c), nn.SiLU(), nn.Conv2d(c, 2 * z_ch if double_z else z_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, ch=128, mult=(1, 2, 4, 4), z_ch=4, n_res=2):
        super().__init__()
        c = ch * mult[-1]
        layers = [nn.Conv2d(z_ch, c, 3, padding=1), ResBlock(c, c)]
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                layers.append(ResBlock(c, ch * m)); c = ch * m
            if i > 0:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]
        layers += [Norm(c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class DiagonalGaussian:
    def __init__(self, params):
        self.mean, logvar = params.chunk(2, 1)
        self.logvar = logvar.clamp(-30, 20)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, g=None):
        return self.mean + self.std * torch.randn(self.mean.shape, generator=g, device=self.mean.device)

    def kl(self):
        return 0.5 * torch.sum(self.mean**2 + self.logvar.exp() - 1 - self.logvar, dim=(1, 2, 3))


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, dim, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1 / n_codes, 1 / n_codes)
        self.beta = beta

    def forward(self, z):
        B, C, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        idx = torch.cdist(flat, self.codebook.weight).argmin(1)
        q = self.codebook(idx).view(B, H, W, C).permute(0, 3, 1, 2)
        loss = F.mse_loss(q, z.detach()) + self.beta * F.mse_loss(z, q.detach())
        return z + (q - z).detach(), loss, idx  # straight-through estimator


class AutoencoderKL(nn.Module):
    def __init__(self, ch=128, mult=(1, 2, 4, 4), z_ch=4, n_res=2):
        super().__init__()
        self.encoder, self.decoder = Encoder(ch, mult, z_ch, True, n_res), Decoder(ch, mult, z_ch, n_res)
        self.quant_conv, self.post_quant_conv = nn.Conv2d(2 * z_ch, 2 * z_ch, 1), nn.Conv2d(z_ch, z_ch, 1)
        self.f = 2 ** (len(mult) - 1)

    def encode(self, x):
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, g=None):
        post = self.encode(x)
        return self.decode(post.sample(g)), post


class AutoencoderVQ(nn.Module):
    def __init__(self, ch=128, mult=(1, 2, 4), z_ch=3, n_codes=8192, n_res=2):
        super().__init__()
        self.encoder, self.decoder = Encoder(ch, mult, z_ch, False, n_res), Decoder(ch, mult, z_ch, n_res)
        self.quantize = VectorQuantizer(n_codes, z_ch)
        self.f = 2 ** (len(mult) - 1)

    def encode(self, x):
        return self.encoder(x)  # LDM trains on pre-quantisation latents; VQ is part of the decoder

    def decode(self, z):
        q, _, _ = self.quantize(z)
        return self.decoder(q)

    def forward(self, x, g=None):
        z = self.encoder(x)
        q, vq_loss, _ = self.quantize(z)
        return self.decoder(q), vq_loss


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (Isola et al.) used for the adversarial loss."""
    def __init__(self, ch=64, n_layers=3):
        super().__init__()
        layers, c = [nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2)], ch
        for i in range(1, n_layers):
            layers += [nn.Conv2d(c, c * 2, 4, 2, 1), nn.GroupNorm(1, c * 2), nn.LeakyReLU(0.2)]; c *= 2
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real, fake):
    return 0.5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())
