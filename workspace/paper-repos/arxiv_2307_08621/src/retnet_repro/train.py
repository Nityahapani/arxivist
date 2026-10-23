"""Parallel (or chunkwise) LM training plus a decoding-cost comparison: recurrent O(1) decoding vs
re-running the parallel form each step (Section 3.4 / Fig. 1 analogue at toy scale)."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import batches, markov_tokens
from .model import RetNet
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def decode_benchmark(model, prompt_len, n_new, vocab, g):
    prompt = torch.randint(0, vocab, (1, prompt_len), generator=g)
    t0 = time.perf_counter()
    model.generate(prompt, n_new)
    rec = time.perf_counter() - t0
    t0 = time.perf_counter()
    ids = prompt
    for _ in range(n_new):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], 1)
    par = time.perf_counter() - t0
    return {"recurrent_tokens_per_s": n_new / rec, "parallel_recompute_tokens_per_s": n_new / par}


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    tokens = markov_tokens(d["n_tokens"], d["vocab"], g)
    model = RetNet(d["vocab"], m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=(0.9, 0.98), weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / t["warmup"], max(0.0, (t["steps"] - s) / max(1, t["steps"] - t["warmup"]))))
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "retnet"), cfg)
    model.train()
    for step in range(1, t["steps"] + 1):
        x, y = batches(tokens, t["seq_len"], t["batch_size"], g)
        logits = model(x.to(device), t.get("mode", "parallel"), t.get("chunk", 64))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("clip", 2.0))
        opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "loss": loss.item()})
    model.eval().cpu()
    bench = decode_benchmark(model, cfg["eval"]["prompt_len"], cfg["eval"]["n_new"], d["vocab"], g)
    return logger.summary({"params": count_params(model), "final_loss": loss.item(), **bench, "is_paper_result": False if smoke else None})
