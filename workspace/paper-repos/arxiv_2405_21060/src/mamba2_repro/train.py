"""Train a Mamba-2 LM on MQAR (associative recall, cf. Fig. 8) with AdamW + warmup-cosine."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import mqar_batch
from .model import Mamba2LM
from .utils import RunLogger, count_params, get_device, set_seed


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    model = Mamba2LM(d["vocab"], m["d_model"], m["n_layers"], m["d_state"], m["head_dim"], m["expand"], m["chunk"]).to(device)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith(("A_log", "D", "dt_bias")) else decay).append(p)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=t["weight_decay"]), dict(params=no_decay, weight_decay=0.0)],
                            lr=t["lr"], betas=(0.9, 0.95))
    S, W = t["steps"], t["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, S - W))))
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "mamba2"), cfg)
    model.train()
    for step in range(1, S + 1):
        x, y = mqar_batch(t["batch_size"], d["n_kv"], d["n_queries"], d["vocab"], g)
        loss = F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == S:
            logger.log({"step": step, "loss": loss.item()})
    model.eval()
    with torch.no_grad():
        x, y = mqar_batch(256, d["n_kv"], d["n_queries"], d["vocab"], g)
        pred = model(x.to(device)).argmax(-1).cpu()
        sel = y != -100
        acc = (pred[sel] == y[sel]).float().mean().item()
    return logger.summary({"params": count_params(model), "final_loss": loss.item(), "mqar_accuracy": acc,
                           "is_paper_result": False if smoke else None})
