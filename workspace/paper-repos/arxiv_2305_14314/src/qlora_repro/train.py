"""QLoRA finetuning pipeline at toy scale:
1. pretrain a tiny GPT in full precision on task A (stand-in for a pretrained LLM);
2. quantize every linear layer to NF4 with double quantization and attach LoRA adapters;
3. finetune only the adapters on task B with the paper's optimiser settings (Adam, constant LR,
   max grad norm 0.3); report NF4 vs FP4 vs Int4 quantization error on the pretrained weights.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import TinyGPT
from .qlora import apply_qlora
from .quant import dequantize_blockwise, fp4_levels, int4_levels, nf4_levels, quantize_blockwise
from .utils import RunLogger, count_params, get_device, set_seed


def task_batch(kind, bs, T, V, g):
    x = torch.randint(0, V, (bs, T + 1), generator=g)
    if kind == "copy":
        x[:, T // 2 + 1:] = x[:, 1:T + 1 - T // 2]
    else:  # "shift": second half is first half shifted by +1 mod V
        x[:, T // 2 + 1:] = (x[:, 1:T + 1 - T // 2] + 1) % V
    return x[:, :-1], x[:, 1:]


def lm_loss(model, x, y, T):
    logits = model(x)[:, T // 2:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, T // 2:].reshape(-1))


def quant_errors(model):
    errs = {}
    for name, levels in (("nf4", nf4_levels()), ("fp4", fp4_levels()), ("int4", int4_levels())):
        se, n = 0.0, 0
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                c, a, s = quantize_blockwise(mod.weight, levels)
                se += (dequantize_blockwise(c, a, s, levels) - mod.weight).pow(2).sum().item()
                n += mod.weight.numel()
        errs[name] = se / n
    return errs


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t, q = cfg["data"], cfg["model"], cfg["train"], cfg["qlora"]
    V, T = d["vocab"], d["seq_len"]
    model = TinyGPT(V, m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], T).to(device)
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "qlora"), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=t["pretrain_lr"])
    for step in range(t["pretrain_steps"]):
        x, y = task_batch("copy", t["batch_size"], T, V, g)
        loss = lm_loss(model, x.to(device), y.to(device), T)
        opt.zero_grad(); loss.backward(); opt.step()
    errs = quant_errors(model)
    qmodel = apply_qlora(copy.deepcopy(model), q["r"], q["alpha"], q["dropout"], block=q["block"], dq_block=q["dq_block"]).to(device)
    trainable = [p for p in qmodel.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=t["lr"], betas=(0.9, 0.999))  # constant LR schedule
    qmodel.train()
    for step in range(1, t["steps"] + 1):
        x, y = task_batch("shift", t["batch_size"], T, V, g)
        loss = lm_loss(qmodel, x.to(device), y.to(device), T)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, t["max_grad_norm"])
        opt.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "loss": loss.item()})
    return logger.summary({"base_params": count_params(model), "trainable_params": sum(p.numel() for p in trainable),
                           "quant_mse": errs, "final_loss": loss.item(), "is_paper_result": False if smoke else None})
