"""ImageNet training recipe (Section 5.2): RMSProp (decay 0.9, momentum 0.9), BN momentum 0.99,
weight decay 1e-5, initial LR 0.256 decayed by 0.97 every 2.4 epochs, SiLU, drop connect 0.2,
dropout increasing with model size; AutoAugment (not implemented here)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .model import SCALING, efficientnet
from .synth_images import synthetic_images
from .utils import RunLogger, count_params, get_device, set_seed


def load(cfg, train, g):
    if cfg["name"] == "synthetic":
        return synthetic_images(cfg["n_train"] if train else cfg["n_test"], cfg["image_size"], cfg["n_classes"], g)
    raise NotImplementedError("Provide an ImageNet loader (see README).")  # pragma: no cover


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    w, dm, res, drop = SCALING[m["name"]]
    if m.get("width_mult"):  # debug override for tiny CPU checks
        w, dm = m["width_mult"], m["depth_mult"]
    from .model import EfficientNet
    model = EfficientNet(w, dm, drop, d["n_classes"], m.get("drop_connect", 0.2)).to(device)
    xtr, ytr = load(d, True, g)
    xte, yte = load(d, False, g)
    opt = torch.optim.RMSprop(model.parameters(), lr=t["lr"] * t["batch_size"] / 4096 if t.get("scale_lr") else t["lr"],
                              alpha=0.9, momentum=0.9, eps=1e-3, weight_decay=t["weight_decay"])
    steps_per_decay = max(1, int(t["decay_every_epochs"] * t["steps_per_epoch"]))
    sched = torch.optim.lr_scheduler.StepLR(opt, steps_per_decay, gamma=t["decay_rate"])
    logger = RunLogger(Path(output_root) / cfg.get("run_name", m["name"]), cfg)
    model.train()
    for step in range(t["steps"]):
        idx = torch.randint(0, len(xtr), (t["batch_size"],), generator=g)
        loss = F.cross_entropy(model(xtr[idx].to(device)), ytr[idx].to(device), label_smoothing=t.get("label_smoothing", 0.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % t["log_every"] == 0 or step + 1 == t["steps"]:
            logger.log({"step": step + 1, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
    model.eval()
    with torch.no_grad():
        acc = (model(xte.to(device)).argmax(1).cpu() == yte).float().mean().item()
    return logger.summary({"model": m["name"], "train_resolution": res, "params": count_params(model), "top1": acc,
                           "is_paper_result": False if smoke else None})
