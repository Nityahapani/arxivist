"""Train at L_train, evaluate perplexity at L_valid >= L_train (extrapolation, Section 4)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import batches, synthetic_stream
from .evaluate import nonoverlapping_ppl
from .model import TransformerLM
from .utils import RunLogger, count_params, get_device, set_seed


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    if d["name"] == "synthetic":
        train_s = synthetic_stream(d["n_train_tokens"], d["vocab"], g)
        valid_s = synthetic_stream(d["n_valid_tokens"], d["vocab"], g)
        vocab = d["vocab"]
    else:  # pragma: no cover
        from .data import wikitext103
        train_s, vocab = wikitext103("train")
        valid_s, _ = wikitext103("validation")
    results = {}
    for pos in m["positions"]:
        set_seed(seed)
        model = TransformerLM(vocab, m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], pos,
                              max_len=max(cfg["eval"]["lengths"]), dropout=m.get("dropout", 0.0)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=t["lr"], betas=(0.9, 0.98))
        logger = RunLogger(Path(output_root) / f"{pos}_Ltrain{t['seq_len']}", cfg)
        it = batches(train_s, t["seq_len"], t["batch_size"], g)
        for step in range(1, t["steps"] + 1):
            lr = t["lr"] * min(1.0, step / max(1, t["warmup"]))
            for gp in opt.param_groups:
                gp["lr"] = lr
            x, y = next(it)
            loss = F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("clip", 1.0))
            opt.step()
            if step % t["log_every"] == 0 or step == t["steps"]:
                logger.log({"step": step, "loss": loss.item()})
        ppl = {str(L): nonoverlapping_ppl(model, valid_s, L, device, cfg["eval"].get("max_windows")) for L in cfg["eval"]["lengths"]}
        results[pos] = logger.summary({"position": pos, "L_train": t["seq_len"], "valid_ppl_by_length": ppl,
                                       "params": count_params(model), "is_paper_result": False if smoke else None})
    return {"runs": results, "is_paper_result": False if smoke else None}
