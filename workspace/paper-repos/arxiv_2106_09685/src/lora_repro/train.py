"""Training entry points.

* ``backend: synthetic`` — CPU-friendly pipeline check: pretrain a small Transformer on one task,
  freeze it, inject LoRA into W_q/W_v and adapt to a second task.
* ``backend: hf`` — the paper's RoBERTa-base GLUE protocol (Table 2 / Table 9) on top of
  Hugging Face ``transformers``; requires the optional ``[hf]`` dependencies and a GPU.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import synthetic_task
from .lora import inject_lora, mark_only_lora_as_trainable, merge_all, unmerge_all
from .model import TinyTransformer
from .utils import RunLogger, count_params, get_device, set_seed


def _accuracy(model, ids, y, bs=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            correct += (model(ids[i:i + bs]).argmax(-1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(ids)


def _fit(model, ids, y, steps, lr, bs, logger, phase, g, eval_every, val):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(ids), (bs,), generator=g)
        loss = F.cross_entropy(model(ids[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            logger.log({"phase": phase, "step": step, "loss": loss.item(), "val_acc": _accuracy(model, *val)})
    return loss.item()


def run_synthetic(cfg: dict, out: Path) -> dict:
    set_seed(cfg.get("seed", 0))
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    m, d, t, l = cfg["model"], cfg["data"], cfg["train"], cfg["lora"]
    model = TinyTransformer(d["vocab"], m["d_model"], m["n_layers"], m["n_heads"], d["seq_len"], n_classes=2)
    logger = RunLogger(out, cfg)
    pre = synthetic_task(d["n_train"], d["seq_len"], d["vocab"], "majority", g)
    pre_val = synthetic_task(d["n_val"], d["seq_len"], d["vocab"], "majority", g)
    _fit(model, *pre, t["pretrain_steps"], t["pretrain_lr"], t["batch_size"], logger, "pretrain", g, t["eval_every"], pre_val)
    total = count_params(model)
    ds = synthetic_task(d["n_train"], d["seq_len"], d["vocab"], "first_half", g)
    val = synthetic_task(d["n_val"], d["seq_len"], d["vocab"], "first_half", g)
    acc_before = _accuracy(model, *val)
    replaced = inject_lora(model, l["target_modules"], r=l["r"], alpha=l["alpha"], dropout=l.get("dropout", 0.0))
    mark_only_lora_as_trainable(model, train_names=("head",))
    trainable = count_params(model, trainable_only=True)
    final_loss = _fit(model, *ds, t["steps"], t["lr"], t["batch_size"], logger, "lora", g, t["eval_every"], val)
    acc_after = _accuracy(model, *val)
    model.eval()
    with torch.no_grad():
        ref = model(val[0][:64])
        merge_all(model)
        merged = model(val[0][:64])
        unmerge_all(model)
    return logger.summary({
        "backend": "synthetic",
        "replaced_modules": len(replaced),
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total,
        "downstream_val_acc_before_lora": acc_before,
        "downstream_val_acc_after_lora": acc_after,
        "final_loss": final_loss,
        "merge_max_abs_diff": (ref - merged).abs().max().item(),
        "is_paper_result": False,
    })


def run_hf_glue(cfg: dict, out: Path) -> dict:  # pragma: no cover - requires transformers + GPU
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, \
        get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader

    from .data import load_glue

    device = get_device(cfg.get("device"))
    task = cfg["task"]
    hp = cfg["glue_hparams"][task]
    set_seed(cfg.get("seed", 0))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    n_labels = 1 if task == "stsb" else (3 if task == "mnli" else 2)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["base_model"], num_labels=n_labels)
    l = cfg["lora"]
    inject_lora(model, l["target_modules"], r=l["r"], alpha=l["alpha"], dropout=l.get("dropout", 0.0),
                init=l.get("init", "gaussian"))
    mark_only_lora_as_trainable(model, train_names=("classifier",))
    model.to(device)
    train_ds, val_ds = load_glue(task, tok, cfg.get("max_seq_len", 512))
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(columns=cols)
    val_ds.set_format(columns=cols)
    coll = DataCollatorWithPadding(tok)
    tl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=coll)
    vl = DataLoader(val_ds, batch_size=64, collate_fn=coll)
    steps = len(tl) * hp["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg.get("warmup_ratio", 0.06) * steps), steps)
    logger = RunLogger(out / f"{task}_seed{cfg.get('seed', 0)}", cfg)
    best = -math.inf
    for epoch in range(hp["epochs"]):
        model.train()
        for batch in tl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        metric = evaluate_glue(model, vl, task, device)
        best = max(best, metric)
        logger.log({"epoch": epoch, "loss": loss.item(), "metric": metric})
    return logger.summary({"task": task, "best_metric": best, "trainable_params": count_params(model, True)})


def evaluate_glue(model, loader, task, device):  # pragma: no cover
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds.append(logits.squeeze(-1) if task == "stsb" else logits.argmax(-1))
            labels.append(batch["labels"])
    p, y = torch.cat(preds).float().cpu(), torch.cat(labels).float().cpu()
    if task == "stsb":
        return torch.corrcoef(torch.stack([p, y]))[0, 1].item()
    if task == "cola":
        tp = ((p == 1) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
        fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
        den = torch.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).float()).clamp_min(1e-12)
        return ((tp * tn - fp * fn) / den).item()
    return (p == y).float().mean().item()


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    backend = cfg.get("backend", "synthetic")
    out = Path(output_root) / cfg.get("run_name", backend)
    if backend == "synthetic":
        return run_synthetic(cfg, out)
    if backend == "hf":
        return run_hf_glue(cfg, out)
    raise ValueError(f"unknown backend {backend!r}")
