"""Prior fitting (Algorithm 1): repeatedly sample a dataset from the prior, split into train/test,
minimise cross-entropy of test labels given the training set. Then evaluate the PFN zero-shot on a
real dataset (sklearn breast cancer) against logistic regression."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .model import TabPFN, predict_proba
from .prior import sample_prior_batch
from .utils import RunLogger, count_params, get_device, set_seed


def evaluate_real(model, cfg, g):
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=cfg["n_train"], test_size=cfg["n_test"], random_state=0, stratify=y)
    model.eval()
    p = predict_proba(model, torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), 2, cfg["n_ensemble"], g, cfg["max_features"])
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    return {"tabpfn_auc": float(roc_auc_score(yte, p[:, 1].numpy())), "tabpfn_acc": float((p.argmax(1).numpy() == yte).mean()),
            "logreg_auc": float(roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:, 1]))}


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    p, m, t = cfg["prior"], cfg["model"], cfg["train"]
    model = TabPFN(p["max_features"], p["max_classes"], m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    S, W = t["steps"], t["warmup"]
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / max(1, S - W))))
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "tabpfn"), cfg)
    model.train()
    for step in range(1, S + 1):
        X, y, d, c = sample_prior_batch(t["batch_size"], p["n_samples"], p["max_features"], p["max_classes"], g, p["p_scm"])
        n_tr = int(torch.randint(1, p["n_samples"] - 1, (1,), generator=g))  # random train/test split position
        X, y = X.to(device), y.to(device)
        logits = model(X[:, :n_tr], y[:, :n_tr], X[:, n_tr:])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, n_tr:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == S:
            logger.log({"step": step, "prior_nll": loss.item()})
    ev = evaluate_real(model.cpu(), {**cfg["eval"], "max_features": p["max_features"]}, g)
    return logger.summary({"params": count_params(model), "final_prior_nll": loss.item(), **ev, "is_paper_result": False if smoke else None})
