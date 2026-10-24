"""Train DeepONet / FNN on an operator dataset with Adam + MSE (Section 4)."""
from __future__ import annotations

from pathlib import Path

import torch

from .data import make_dataset
from .model import build
from .utils import RunLogger, count_params, get_device, set_seed


def _t(a, device):
    return torch.from_numpy(a).to(device)


def train_one(cfg: dict, model_cfg: dict, out: Path) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = get_device(cfg.get("device"))
    d, t = cfg["data"], cfg["train"]
    common = dict(m=d["m"], length_scale=d["length_scale"], n_grid=d["n_grid"])
    tr = make_dataset(d["operator"], d["n_train_functions"], d["points_per_function"], seed=seed, **common)
    te = make_dataset(d["operator"], d["n_test_functions"], d["test_points_per_function"], seed=seed + 10_000, **common)
    U, Y, S = (_t(a, device) for a in (tr.u_sensors, tr.y, tr.s))
    Ut, Yt, St = (_t(a, device) for a in (te.u_sensors, te.y, te.s))
    model = build(model_cfg, d["m"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    logger = RunLogger(out, {**cfg, "model": model_cfg})
    g = torch.Generator().manual_seed(seed)
    bs = t["batch_size"] if t["batch_size"] > 0 else len(U)
    for step in range(1, t["steps"] + 1):
        idx = torch.randint(0, len(U), (bs,), generator=g)
        loss = torch.mean((model(U[idx], Y[idx]) - S[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t["eval_every"] == 0 or step == t["steps"]:
            with torch.no_grad():
                test_mse = torch.mean((model(Ut, Yt) - St) ** 2).item()
            logger.log({"step": step, "train_mse": loss.item(), "test_mse": test_mse})
    return logger.summary({"operator": d["operator"], "model": model_cfg.get("kind"), "params": count_params(model),
                           "final_train_mse": loss.item(), "final_test_mse": test_mse,
                           "is_paper_result": False if cfg.get("smoke") else None})


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    cfg = {**cfg, "smoke": smoke}
    results = {}
    for mc in cfg["models"]:
        name = mc.get("name", mc.get("kind"))
        results[name] = train_one(cfg, mc, Path(output_root) / f"{cfg['data']['operator']}_{name}_seed{cfg.get('seed', 0)}")
    first = next(iter(results.values()))
    return {"runs": results, "final_test_mse": first["final_test_mse"], "is_paper_result": False if smoke else None}
