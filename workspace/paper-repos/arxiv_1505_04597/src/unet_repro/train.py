"""Training: SGD with momentum 0.99, large tiles with batch size 1, weighted pixel-wise cross-entropy,
elastic deformation (Section 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .augment import elastic_deform
from .data import synthetic_cells
from .infer import overlap_tile_predict
from .model import UNet, center_crop, output_size
from .utils import RunLogger, count_params, get_device, set_seed
from .weights import unet_weight_map


def iou(pred, target):
    inter = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()
    return inter / max(1, union)


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    model = UNet(1, 2, m["base"], m["depth"], m.get("dropout", 0.5)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"])
    tile = d["tile"]
    out_sz = output_size(tile, m["depth"])
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "unet"), cfg)
    model.train()
    for step in range(1, t["steps"] + 1):
        img, lab, inst = synthetic_cells(tile, d["n_cells"], rng)
        if t.get("elastic", True):
            img, inst = elastic_deform(img, inst, rng=rng)
            lab = (inst > 0).astype(np.int64)
        wmap = unet_weight_map(inst, t["w0"], t["sigma"])
        x = torch.from_numpy(img[None]).float().to(device)
        y = center_crop(torch.from_numpy(lab)[None], out_sz, out_sz).to(device)
        w = center_crop(torch.from_numpy(wmap)[None].float(), out_sz, out_sz).to(device)
        logits = model(x)
        loss = (F.cross_entropy(logits, y, reduction="none") * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "weighted_ce": loss.item()})
    img, lab, _ = synthetic_cells(d["eval_size"], d["n_cells"], rng)
    pred = overlap_tile_predict(model.cpu(), img, tile, m["depth"]).argmax(0)
    return logger.summary({"params": count_params(model), "output_size": out_sz, "eval_iou": float(iou(pred, lab)),
                           "is_paper_result": False if smoke else None})
