"""BYOL pretraining (Algorithm 1) with LARS + warmup-cosine and the linear evaluation protocol."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .augment import BYOLAugment
from .data import load_dataset, normalize
from .lars import LARS, lars_param_groups, warmup_cosine
from .loss import byol_loss
from .model import BYOL
from .schedule import tau_schedule
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def extract(encoder, x, device, bs=256):
    encoder.eval()
    return torch.cat([encoder(normalize(x[i:i + bs].to(device))).cpu() for i in range(0, len(x), bs)])


def linear_eval(encoder, train, test, device, cfg, g):
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
    train, test = load_dataset(d, True, g), load_dataset(d, False, g)
    model = BYOL(m["arch"], m.get("width_mult", 1), m.get("cifar_stem", False), m.get("base_width", 64),
                 m["hidden"], m["proj_dim"]).to(device)
    aug1, aug2 = BYOLAugment(d["image_size"], "T", t["blur"]), BYOLAugment(d["image_size"], "T_prime", t["blur"])
    bs, steps = t["batch_size"], t["steps"]
    base_lr = t["lr_per_256"] * bs / 256
    online = torch.nn.ModuleDict(dict(e=model.encoder, p=model.projector, q=model.predictor))
    opt = LARS(lars_param_groups(online, t["weight_decay"]), lr=base_lr)
    warm = int(t.get("warmup_fraction", 0.01) * steps)
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "byol"), cfg)
    model.train()
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = warmup_cosine(step, steps, warm, base_lr)
        x = train[0][torch.randint(0, len(train[0]), (bs,), generator=g)]
        v1, v2 = normalize(aug1(x, g)).to(device), normalize(aug2(x, g)).to(device)
        loss = byol_loss(model.online(v1), model.online(v2), model.target(v1), model.target(v2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tau = tau_schedule(step, steps, t["tau_base"])
        model.update_target(tau)
        if (step + 1) % t["log_every"] == 0 or step + 1 == steps:
            logger.log({"step": step + 1, "loss": loss.item(), "lr": opt.param_groups[0]["lr"], "tau": tau})
    acc = linear_eval(model.encoder, train, test, device, cfg["linear_eval"], g)
    return logger.summary({"final_byol_loss": loss.item(), "linear_eval_top1": acc,
                           "encoder_params": count_params(model.encoder), "is_paper_result": False if smoke else None})
