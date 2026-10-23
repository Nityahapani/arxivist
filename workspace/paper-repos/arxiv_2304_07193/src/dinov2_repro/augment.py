"""DINOv2 multi-crop augmentation (Section 4; following DINO), on batched tensors in [0, 1].

2 global crops (scale 0.32-1) and N local crops (scale 0.05-0.32) per image. Global crop 1: blur p=1.0;
global crop 2: blur p=0.1 and solarize p=0.2; local crops: blur p=0.5. All: flip, colour jitter p=0.8
(0.4, 0.4, 0.2, 0.1), grayscale p=0.2.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def random_resized_crop(x: torch.Tensor, size: int, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), g=None) -> torch.Tensor:
    B, C, H, W = x.shape
    out = torch.empty(B, C, size, size, dtype=x.dtype, device=x.device)
    area = H * W
    for i in range(B):
        for _ in range(10):
            target = area * torch.empty(1).uniform_(*scale, generator=g).item()
            logr = torch.empty(1).uniform_(math.log(ratio[0]), math.log(ratio[1]), generator=g).item()
            ar = math.exp(logr)
            w = int(round(math.sqrt(target * ar)))
            h = int(round(math.sqrt(target / ar)))
            if 0 < w <= W and 0 < h <= H:
                break
        else:
            h, w = H, W
        top = torch.randint(0, H - h + 1, (1,), generator=g).item()
        left = torch.randint(0, W - w + 1, (1,), generator=g).item()
        crop = x[i:i + 1, :, top:top + h, left:left + w]
        out[i] = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)[0]
    return out


def hflip(x, p=0.5, g=None):
    m = torch.rand(x.size(0), generator=g) < p
    x = x.clone()
    x[m] = x[m].flip(-1)
    return x


def rgb_to_hsv(x):
    r, gch, b = x.unbind(1)
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    dc = delta.clamp_min(1e-8)
    rc, gc, bc = (maxc - r) / dc, (maxc - gch) / dc, (maxc - b) / dc
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == gch, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, v], 1)


def hsv_to_rgb(x):
    h, s, v = x.unbind(1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.long() % 6
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = torch.zeros_like(x)
    for k, (a, b_, c) in enumerate(sel):
        m = (i == k).unsqueeze(1)
        out = torch.where(m, torch.stack([a, b_, c], 1), out)
    return out


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def color_jitter(x, s=1.0, p=0.8, g=None, strengths=None):
    """Random-order colour jitter. Default strengths (0.8s, 0.8s, 0.8s, 0.2s) follow SimCLR; BYOL passes
    brightness 0.4, contrast 0.4, saturation 0.2, hue 0.1 (BYOL Appendix B, Table 6)."""
    sb, sc, ss, sh = strengths or (0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    B = x.size(0)
    apply = (torch.rand(B, generator=g) < p).view(B, 1, 1, 1)
    y = x
    def u(lo, hi):
        return torch.empty(B, 1, 1, 1).uniform_(lo, hi, generator=g)
    ops = torch.randperm(4, generator=g).tolist()
    for op in ops:
        if op == 0:
            y = (y * u(max(0, 1 - sb), 1 + sb)).clamp(0, 1)
        elif op == 1:
            mean = _gray(y).mean((2, 3), keepdim=True)
            y = ((y - mean) * u(max(0, 1 - sc), 1 + sc) + mean).clamp(0, 1)
        elif op == 2:
            gr = _gray(y)
            y = ((y - gr) * u(max(0, 1 - ss), 1 + ss) + gr).clamp(0, 1)
        else:
            hsv = rgb_to_hsv(y)
            hsv[:, 0] = (hsv[:, 0] + u(-sh, sh)[:, 0]) % 1.0
            y = hsv_to_rgb(hsv).clamp(0, 1)
    return torch.where(apply, y, x)


def random_grayscale(x, p=0.2, g=None):
    m = (torch.rand(x.size(0), generator=g) < p).view(-1, 1, 1, 1)
    return torch.where(m, _gray(x).expand_as(x), x)


def gaussian_blur(x, p=0.5, g=None, sigma=(0.1, 2.0)):
    """Kernel size 10% of image side; sigma ~ U[0.1, 2.0] (Appendix A)."""
    B, C, H, W = x.shape
    k = max(3, int(0.1 * H) // 2 * 2 + 1)
    out = x.clone()
    for i in range(B):
        if torch.rand(1, generator=g).item() < p:
            sg = torch.empty(1).uniform_(*sigma, generator=g).item()
            r = torch.arange(k, dtype=x.dtype) - k // 2
            ker = torch.exp(-r**2 / (2 * sg**2))
            ker = ker / ker.sum()
            xi = x[i:i + 1]
            xi = F.conv2d(F.pad(xi, (k // 2, k // 2, 0, 0), mode="reflect"), ker.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
            xi = F.conv2d(F.pad(xi, (0, 0, k // 2, k // 2), mode="reflect"), ker.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
            out[i] = xi[0]
    return out


def solarize(x, p=0.2, g=None, threshold=0.5):
    """Pixels above the threshold are inverted (x -> 1 - x)."""
    m = (torch.rand(x.size(0), generator=g) < p).view(-1, 1, 1, 1)
    return torch.where(m, torch.where(x < threshold, x, 1.0 - x), x)


class MultiCrop:
    def __init__(self, global_size=224, local_size=98, n_local=8, global_scale=(0.32, 1.0), local_scale=(0.05, 0.32), blur=True):
        self.gs, self.ls, self.n_local = global_size, local_size, n_local
        self.gscale, self.lscale, self.blur = global_scale, local_scale, blur

    def _base(self, x, size, scale, g):
        x = random_resized_crop(x, size, scale=scale, g=g)
        x = hflip(x, g=g)
        x = color_jitter(x, p=0.8, g=g, strengths=(0.4, 0.4, 0.2, 0.1))
        return random_grayscale(x, p=0.2, g=g)

    def __call__(self, x, g=None):
        g1 = self._base(x, self.gs, self.gscale, g)
        g2 = self._base(x, self.gs, self.gscale, g)
        if self.blur:
            g1, g2 = gaussian_blur(g1, p=1.0, g=g), gaussian_blur(g2, p=0.1, g=g)
        g2 = solarize(g2, p=0.2, g=g)
        locs = []
        for _ in range(self.n_local):
            l = self._base(x, self.ls, self.lscale, g)
            locs.append(gaussian_blur(l, p=0.5, g=g) if self.blur else l)
        return [g1, g2], locs
