"""Diffusion in latent space (Section 3.2, Eq. 2): L_LDM = E ||eps - eps_theta(z_t, t)||^2 with the DDPM
forward process, the "scaled linear" beta schedule (beta 0.00085 -> 0.012, T = 1000) used by the LDM
configs, DDIM sampling (Song et al.) and classifier-free guidance (Ho & Salimans):
eps_hat = eps(z, t, null) + s * (eps(z, t, c) - eps(z, t, null))."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Schedule:
    def __init__(self, T=1000, beta_start=0.00085, beta_end=0.012, kind="scaled_linear"):
        if kind == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, T, dtype=torch.float64) ** 2
        else:
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        self.T, self.betas = T, betas
        self.alphas_cumprod = torch.cumprod(1 - betas, 0)

    def q_sample(self, z0, t, noise):
        a = self.alphas_cumprod.to(z0.device)[t].float().view(-1, 1, 1, 1)
        return a.sqrt() * z0 + (1 - a).sqrt() * noise


def ldm_loss(model, sched, z0, g=None, **cond):
    t = torch.randint(0, sched.T, (z0.size(0),), generator=g).to(z0.device)
    noise = torch.randn(z0.shape, generator=g).to(z0.device)
    return F.mse_loss(model(sched.q_sample(z0, t, noise), t, **cond), noise)


def guided_eps(model, z, t, cond, uncond, scale):
    if scale == 1.0 or uncond is None:
        return model(z, t, **cond)
    return model(z, t, **uncond) + scale * (model(z, t, **cond) - model(z, t, **uncond))


@torch.no_grad()
def ddim_sample(model, sched, shape, steps=50, eta=0.0, cond=None, uncond=None, guidance=1.0, g=None, z=None):
    cond = cond or {}
    z = torch.randn(shape, generator=g) if z is None else z
    ts = torch.linspace(sched.T - 1, 0, steps).round().long()
    ac = sched.alphas_cumprod.float()
    for i, t in enumerate(ts):
        a_t = ac[t]
        a_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        eps = guided_eps(model, z, t.repeat(shape[0]), cond, uncond, guidance)
        z0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
        noise = torch.randn(shape, generator=g) if eta > 0 else 0.0
        z = a_prev.sqrt() * z0 + (1 - a_prev - sigma**2).clamp_min(0).sqrt() * eps + sigma * noise
    return z
