"""MAE pretraining (Section 4 / Appendix A.1) and a linear probe on frozen features."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .mae import MaskedAutoencoder
from .synth_images import synthetic_images
from .utils import RunLogger, count_params, get_device, set_seed


def load(cfg, train, g):
    if cfg["name"] == "synthetic":
        return synthetic_images(cfg["n_train"] if train else cfg["n_test"], cfg["image_size"], cfg["n_classes"], g)
    if cfg["name"] == "imagefolder":  # pragma: no cover - ImageNet-1K in torchvision ImageFolder layout
        raise NotImplementedError("Use a streaming ImageNet loader; see README (Paper-scale protocol).")
    raise ValueError(cfg["name"])


def random_resized_crop_flip(x, size, g, scale=(0.2, 1.0)):
    """MAE pretraining augmentation: RandomResizedCrop(scale=(0.2, 1.0)) + horizontal flip only."""
    B, C, H, W = x.shape
    out = torch.empty(B, C, size, size)
    for i in range(B):
        s = torch.empty(1).uniform_(*scale, generator=g).item()
        side = max(1, int(round(math.sqrt(s) * H)))
        t = torch.randint(0, H - side + 1, (1,), generator=g).item()
        l = torch.randint(0, W - side + 1, (1,), generator=g).item()
        crop = F.interpolate(x[i:i + 1, :, t:t + side, l:l + side], size=(size, size), mode="bicubic", align_corners=False)[0]
        out[i] = crop.flip(-1) if torch.rand(1, generator=g).item() < 0.5 else crop
    return out


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(1, warmup)
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    xtr, ytr = load(d, True, g)
    xte, yte = load(d, False, g)
    model = MaskedAutoencoder(d["image_size"], 3, m["arch"], m["dec_dim"], m["dec_depth"], m["dec_heads"], m["norm_pix_loss"]).to(device)
    bs = t["batch_size"]
    base_lr = t["blr"] * bs / 256  # absolute lr = base lr x batch / 256 (Appendix A.1)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim == 1 or n.endswith("token") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=base_lr, betas=(0.9, 0.95))
    total, warm = t["steps"], int(t["warmup_fraction"] * t["steps"])
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "mae"), cfg)
    model.train()
    for step in range(total):
        for gp in opt.param_groups:
            gp["lr"] = cosine_lr(step, total, warm, base_lr)
        idx = torch.randint(0, len(xtr), (bs,), generator=g)
        x = random_resized_crop_flip(xtr[idx], d["image_size"], g).to(device)
        loss, _, mask = model(x, t["mask_ratio"], g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % t["log_every"] == 0 or step + 1 == total:
            logger.log({"step": step + 1, "loss": loss.item(), "mask_fraction": mask.mean().item()})
    model.eval()
    ftr, fte = model.features(xtr.to(device)).cpu(), model.features(xte.to(device)).cpu()
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6  # BatchNorm-without-affine before the linear probe (Appendix A.3)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    clf = torch.nn.Linear(ftr.size(1), int(ytr.max()) + 1)
    popt = torch.optim.SGD(clf.parameters(), lr=0.1, momentum=0.9)
    for _ in range(cfg["linear_probe"]["steps"]):
        lossp = F.cross_entropy(clf(ftr), ytr)
        popt.zero_grad(); lossp.backward(); popt.step()
    acc = (clf(fte).argmax(1) == yte).float().mean().item()
    return logger.summary({"final_recon_loss": loss.item(), "linear_probe_top1": acc,
                           "encoder_params": count_params(model.blocks) + count_params(model.patch_embed),
                           "is_paper_result": False if smoke else None})
