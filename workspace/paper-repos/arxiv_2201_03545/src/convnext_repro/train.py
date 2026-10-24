"""ImageNet-style supervised training recipe (AdamW, cosine, Mixup/CutMix, label smoothing,
stochastic depth, layer scale, EMA)."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from .augment import mixup_cutmix, soft_ce
from .model import ConvNeXt
from .synth_images import synthetic_images
from .utils import RunLogger, count_params, get_device, set_seed


def load(cfg, train, g):
    if cfg["name"] == "synthetic":
        return synthetic_images(cfg["n_train"] if train else cfg["n_test"], cfg["image_size"], cfg["n_classes"], g)
    raise NotImplementedError("Provide an ImageNet-1K loader (see README).")  # pragma: no cover


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    xtr, ytr = load(d, True, g)
    xte, yte = load(d, False, g)
    model = ConvNeXt(m["variant"], 3, d["n_classes"], m["drop_path"], m["layer_scale_init"]).to(device)
    ema = copy.deepcopy(model).eval() if t.get("ema_decay") else None
    bs = t["batch_size"]
    lr = t["lr"] * bs / 4096 if t.get("scale_lr", True) else t["lr"]
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    total, warm = t["steps"], int(t["warmup_fraction"] * t["steps"])
    logger = RunLogger(Path(output_root) / cfg.get("run_name", m["variant"]), cfg)
    model.train()
    for step in range(total):
        cur = lr * step / max(1, warm) if step < warm else lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for gp in opt.param_groups:
            gp["lr"] = cur
        idx = torch.randint(0, len(xtr), (bs,), generator=g)
        x, y = mixup_cutmix(xtr[idx], ytr[idx], d["n_classes"], t["mixup"], t["cutmix"], 0.5, t["label_smoothing"], rng)
        loss = soft_ce(model(x.to(device)), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(t["ema_decay"]).add_(p.detach(), alpha=1 - t["ema_decay"])
        if (step + 1) % t["log_every"] == 0 or step + 1 == total:
            logger.log({"step": step + 1, "loss": loss.item(), "lr": cur})
    model.eval()
    with torch.no_grad():
        acc = (model(xte.to(device)).argmax(1).cpu() == yte).float().mean().item()
    return logger.summary({"variant": m["variant"], "params": count_params(model), "top1": acc, "is_paper_result": False if smoke else None})
