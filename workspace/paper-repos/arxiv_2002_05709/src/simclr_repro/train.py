"""SimCLR pretraining (Algorithm 1) followed by the linear evaluation protocol (Section 2.3)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .augment import SimCLRAugment
from .data import load_dataset, normalize
from .lars import LARS, lars_param_groups, warmup_cosine
from .loss import nt_xent
from .model import SimCLR
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def extract(encoder, x, device, bs=256):
    encoder.eval()
    feats = [encoder(normalize(x[i:i + bs].to(device))).cpu() for i in range(0, len(x), bs)]
    return torch.cat(feats)


def linear_eval(encoder, train, test, device, cfg, g):
    """Train a linear classifier on frozen features; report top-1 test accuracy."""
    ftr, fte = extract(encoder, train[0], device), extract(encoder, test[0], device)
    clf = torch.nn.Linear(ftr.size(1), int(train[1].max()) + 1)
    opt = torch.optim.SGD(clf.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True)
    for _ in range(cfg["steps"]):
        idx = torch.randint(0, len(ftr), (min(cfg["batch_size"], len(ftr)),), generator=g)
        loss = F.cross_entropy(clf(ftr[idx]), train[1][idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (clf(fte).argmax(1) == test[1]).float().mean().item()


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    train = load_dataset(d, True, g)
    test = load_dataset(d, False, g)
    model = SimCLR(m["arch"], m.get("width_mult", 1), m["proj_dim"], m.get("cifar_stem", True), m.get("base_width", 64)).to(device)
    aug = SimCLRAugment(d["image_size"], t["color_strength"], blur=t["blur"])
    bs = t["batch_size"]
    base_lr = t["lr_per_256"] * bs / 256
    opt = LARS(lars_param_groups(model, t["weight_decay"]), lr=base_lr)
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "simclr"), cfg)
    steps = t["steps"]
    warm = int(t.get("warmup_fraction", 0.1) * steps)
    model.train()
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = warmup_cosine(step, steps, warm, base_lr)
        idx = torch.randint(0, len(train[0]), (bs,), generator=g)
        x = train[0][idx]
        v1, v2 = normalize(aug(x, g)).to(device), normalize(aug(x, g)).to(device)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = nt_xent(z1, z2, t["temperature"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % t["log_every"] == 0 or step + 1 == steps:
            logger.log({"step": step + 1, "loss": loss.item(), "lr": opt.param_groups[0]["lr"]})
    acc = linear_eval(model.encoder, train, test, device, cfg["linear_eval"], g)
    return logger.summary({"final_contrastive_loss": loss.item(), "linear_eval_top1": acc,
                           "encoder_params": count_params(model.encoder), "is_paper_result": False if smoke else None})
