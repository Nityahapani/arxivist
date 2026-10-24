"""Masked-LM pretraining comparison of RoPE vs absolute sinusoidal positions (cf. Fig. 3 pretraining
curves), on synthetic periodic sequences in smoke runs."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import mask_tokens, periodic_sequences
from .model import RoFormer
from .utils import RunLogger, count_params, get_device, set_seed


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "roformer"), cfg)
    results = {}
    for pos in m["positions"]:
        set_seed(seed)
        g = torch.Generator().manual_seed(seed)
        model = RoFormer(d["vocab"], m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], pos, m.get("linear", False), m.get("dropout", 0.1)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, t["warmup"])))
        model.train()
        for step in range(1, t["steps"] + 1):
            x, y = mask_tokens(periodic_sequences(t["batch_size"], t["seq_len"], d["vocab"], g), g)
            loss = F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten(), ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if step % t["log_every"] == 0 or step == t["steps"]:
                logger.log({"pos": pos, "step": step, "mlm_loss": loss.item()})
        model.eval()
        with torch.no_grad():
            x, y = mask_tokens(periodic_sequences(256, t["seq_len"], d["vocab"], g), g)
            logits = model(x.to(device))
            sel = y != -100
            acc = (logits.argmax(-1).cpu()[sel] == y[sel]).float().mean().item()
        results[pos] = {"final_mlm_loss": loss.item(), "masked_accuracy": acc, "params": count_params(model)}
    return logger.summary({"results": results, "is_paper_result": False if smoke else None})
