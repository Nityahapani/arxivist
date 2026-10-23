"""Train the same GPT with FlashAttention and with standard attention from identical initialisation and
compare losses (FlashAttention is exact, Section 3), plus report the analytical HBM-access ratio."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .io import flash_hbm_accesses, standard_hbm_accesses
from .model import GPT
from .utils import RunLogger, count_params, get_device, set_seed


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "flash"), cfg)
    losses = {}
    for impl in ("flash", "standard"):
        set_seed(seed)
        g = torch.Generator().manual_seed(seed)
        model = GPT(d["vocab"], m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], t["seq_len"], impl, m["br"], m["bc"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        for step in range(1, t["steps"] + 1):
            x = torch.randint(0, d["vocab"], (t["batch_size"], t["seq_len"] + 1), generator=g)
            x[:, 1::2] = x[:, 0:-1:2]  # learnable copy structure
            logits = model(x[:, :-1].to(device))
            loss = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].to(device).flatten())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % t["log_every"] == 0 or step == t["steps"]:
                logger.log({"impl": impl, "step": step, "loss": loss.item()})
        losses[impl] = loss.item()
    N, hd, M = cfg["io"]["seq_len"], cfg["io"]["head_dim"], cfg["io"]["sram_elements"]
    return logger.summary({"params": count_params(model), "final_loss": losses, "loss_abs_diff": abs(losses["flash"] - losses["standard"]),
                           "hbm_access_ratio_standard_over_flash": standard_hbm_accesses(N, hd) / flash_hbm_accesses(N, hd, M),
                           "is_paper_result": False if smoke else None})
