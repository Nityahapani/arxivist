"""Two-stage LDM training (Section 3):
1. train a KL-regularised autoencoder (L1 + KL + PatchGAN after `disc_start`);
2. freeze it, rescale latents by 1/std (Appendix G, 'scale factor'), train a class-conditional latent UNet
   with label dropout (for classifier-free guidance) on epsilon prediction;
3. sample with DDIM + guidance and decode.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .autoencoder import AutoencoderKL, PatchDiscriminator, hinge_d_loss
from .conditioning import ClassEmbedder
from .diffusion import Schedule, ddim_sample, ldm_loss
from .synth_images import synthetic_images
from .unet import UNet
from .utils import RunLogger, count_params, get_device, set_seed


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, a, u, t = cfg["data"], cfg["autoencoder"], cfg["unet"], cfg["train"]
    x_all, y_all = synthetic_images(d["n_train"], d["image_size"], d["n_classes"], g)
    x_all = x_all * 2 - 1
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "ldm"), cfg)

    ae = AutoencoderKL(a["ch"], tuple(a["mult"]), a["z_ch"], a["n_res"]).to(device)
    disc = PatchDiscriminator(a["disc_ch"], 2).to(device)
    opt_ae, opt_d = torch.optim.Adam(ae.parameters(), lr=t["ae_lr"], betas=(0.5, 0.9)), torch.optim.Adam(disc.parameters(), lr=t["ae_lr"], betas=(0.5, 0.9))
    for step in range(1, t["ae_steps"] + 1):
        x = x_all[torch.randint(0, len(x_all), (t["batch_size"],), generator=g)].to(device)
        rec, post = ae(x, g)
        loss = F.l1_loss(rec, x) + a["kl_weight"] * post.kl().mean() / x[0].numel()
        use_gan = step > t["disc_start"]
        if use_gan:
            loss = loss + a["disc_weight"] * (-disc(rec).mean())
        opt_ae.zero_grad(set_to_none=True); loss.backward(); opt_ae.step()
        if use_gan:
            dl = hinge_d_loss(disc(x), disc(rec.detach()))
            opt_d.zero_grad(set_to_none=True); dl.backward(); opt_d.step()
        if step % t["log_every"] == 0 or step == t["ae_steps"]:
            logger.log({"stage": "autoencoder", "step": step, "loss": loss.item()})
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        z_all = torch.cat([ae.encode(x_all[i:i + 64].to(device)).sample(g) for i in range(0, len(x_all), 64)])
    scale = 1.0 / z_all.flatten().std().item()
    z_all = z_all * scale

    emb = ClassEmbedder(d["n_classes"], u["d_ctx"]).to(device)
    unet = UNet(a["z_ch"], u["ch"], tuple(u["mult"]), tuple(u["attn_levels"]), u["n_res"], u["heads"], u["d_ctx"]).to(device)
    sched = Schedule(t["T"])
    opt = torch.optim.AdamW(list(unet.parameters()) + list(emb.parameters()), lr=t["lr"])
    for step in range(1, t["steps"] + 1):
        idx = torch.randint(0, len(z_all), (t["batch_size"],), generator=g)
        y = y_all[idx].clone()
        y[torch.rand(len(y), generator=g) < t["p_uncond"]] = emb.null  # label dropout for CFG
        loss = ldm_loss(unet, sched, z_all[idx], g, context=emb(y.to(device)))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"stage": "ldm", "step": step, "loss": loss.item()})
    unet.eval()
    n = 4
    y = torch.arange(n) % d["n_classes"]
    z = ddim_sample(unet, sched, (n, a["z_ch"], *z_all.shape[-2:]), t["ddim_steps"], 0.0, {"context": emb(y)},
                    {"context": emb(torch.full_like(y, emb.null))}, t["guidance"], g)
    samples = ae.decode(z / scale).clamp(-1, 1)
    return logger.summary({"ae_params": count_params(ae), "unet_params": count_params(unet), "latent_scale": scale,
                           "downsampling_f": ae.f, "final_ldm_loss": loss.item(), "sample_shape": list(samples.shape),
                           "is_paper_result": False if smoke else None})
