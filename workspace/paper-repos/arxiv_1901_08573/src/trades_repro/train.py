"""Adversarial training with the TRADES loss and PGD-20 robust evaluation (Section 5.2)."""
from __future__ import annotations

from pathlib import Path

import torch

from .data import augment, load
from .model import WideResNet
from .trades import accuracy, pgd_attack, trades_loss
from .utils import RunLogger, count_params, get_device, set_seed


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t, a, e = cfg["data"], cfg["model"], cfg["train"], cfg["attack_train"], cfg["attack_eval"]
    xtr, ytr = load(d, True, g)
    xte, yte = load(d, False, g)
    model = WideResNet(m["depth"], m["widen"], m["n_classes"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"])
    steps_per_epoch = max(1, len(xtr) // t["batch_size"])
    total = t.get("steps") or steps_per_epoch * t["epochs"]
    milestones = [int(f * total) for f in t["lr_decay_at"]]
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "trades"), cfg)
    model.train()
    for step in range(total):
        if step in milestones:
            for gp in opt.param_groups:
                gp["lr"] *= 0.1
        idx = torch.randint(0, len(xtr), (t["batch_size"],), generator=g)
        x = augment(xtr[idx], g) if d.get("augment", True) else xtr[idx]
        x, y = x.to(device), ytr[idx].to(device)
        loss, nat, rob = trades_loss(model, x, y, beta=t["beta"], **a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % t["log_every"] == 0 or step + 1 == total:
            logger.log({"step": step + 1, "loss": loss.item(), "natural_ce": nat.item(), "robust_kl": rob.item()})
    n = min(len(xte), e.get("n_eval", len(xte)))
    xe, ye = xte[:n].to(device), yte[:n].to(device)
    nat_acc = accuracy(model, xe, ye)
    rob_acc = accuracy(model, pgd_attack(model, xe, ye, e["eps"], e["step_size"], e["steps"], generator=g), ye)
    return logger.summary({"natural_acc": nat_acc, "robust_acc_pgd": rob_acc, "beta": t["beta"], "params": count_params(model),
                           "steps": total, "is_paper_result": False if smoke else None})
