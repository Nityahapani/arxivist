"""SGD vs SAM training on CIFAR (Section 3.1) with optional m-sharpness and label smoothing."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import augment, load
from .model import WideResNet
from .sam import SAM, sam_step_m_sharpness
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def evaluate(model, x, y, bs=500):
    model.eval()
    correct = sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    model.train()
    return correct / len(x)


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    xtr, ytr = load(d, True, g)
    xte, yte = load(d, False, g)
    model = WideResNet(m["depth"], m["widen"], m["n_classes"]).to(device)
    ls = t.get("label_smoothing", 0.0)
    loss_fn = lambda out, y: F.cross_entropy(out, y, label_smoothing=ls)
    kw = dict(lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"], nesterov=t.get("nesterov", False))
    use_sam = t["optimizer"] == "sam"
    opt = SAM(model.parameters(), torch.optim.SGD, rho=t["rho"], **kw) if use_sam else torch.optim.SGD(model.parameters(), **kw)
    spe = max(1, len(xtr) // t["batch_size"])
    total = t.get("steps") or spe * t["epochs"]
    logger = RunLogger(Path(output_root) / cfg.get("run_name", t["optimizer"]), cfg)
    model.train()
    for step in range(total):
        lr = 0.5 * t["lr"] * (1 + math.cos(math.pi * step / total))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = torch.randint(0, len(xtr), (t["batch_size"],), generator=g)
        x, y = augment(xtr[idx], g).to(device), ytr[idx].to(device)
        if use_sam and t.get("m_sharpness"):
            loss = sam_step_m_sharpness(model, loss_fn, x, y, opt, t["m_sharpness"])
        elif use_sam:
            def closure():
                l = loss_fn(model(x), y)
                l.backward()
                return l
            opt.zero_grad(set_to_none=True)
            loss = opt.step(closure).item()
        else:
            l = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            loss = l.item()
        if (step + 1) % t["log_every"] == 0 or step + 1 == total:
            logger.log({"step": step + 1, "loss": loss, "lr": lr})
    acc = evaluate(model, xte.to(device), yte.to(device))
    return logger.summary({"optimizer": t["optimizer"], "rho": t.get("rho"), "test_acc": acc, "test_error": 1 - acc,
                           "params": count_params(model), "steps": total, "is_paper_result": False if smoke else None})
