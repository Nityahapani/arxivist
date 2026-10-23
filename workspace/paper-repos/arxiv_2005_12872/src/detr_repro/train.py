"""DETR training (Section 4, Appendix A.4): AdamW with lr 1e-4 (transformer) and 1e-5 (backbone),
weight decay 1e-4, gradient clipping 0.1, step LR drop x0.1, auxiliary decoding losses."""
from __future__ import annotations

from pathlib import Path

import torch

from .box_ops import box_iou, cxcywh_to_xyxy
from .criterion import SetCriterion
from .data import synthetic_detection
from .model import DETR
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def recall_at_05(model, imgs, targets, thresh=0.5):
    out = model(imgs)
    logits, boxes = out["logits"][-1], out["boxes"][-1]
    hit, tot = 0, 0
    for b, t in enumerate(targets):
        keep = logits[b].softmax(-1)[:, :-1].max(-1).values > 0.3
        cls = logits[b][:, :-1].argmax(-1)
        if keep.any() and len(t["labels"]):
            iou, _ = box_iou(cxcywh_to_xyxy(t["boxes"]), cxcywh_to_xyxy(boxes[b][keep]))
            same = t["labels"][:, None] == cls[keep][None]
            hit += ((iou > thresh) & same).any(1).sum().item()
        tot += len(t["labels"])
    return hit / max(1, tot)


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    model = DETR(d["n_classes"], m["n_queries"], m["d_model"], m["heads"], m["enc_layers"], m["dec_layers"], m["d_ff"],
                 m["dropout"], m["arch"], m["base_width"]).to(device)
    crit = SetCriterion(d["n_classes"], t["eos_coef"], 1.0, t["w_l1"], t["w_giou"]).to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad]
    opt = torch.optim.AdamW([dict(params=rest, lr=t["lr"]), dict(params=bb, lr=t["lr_backbone"])], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, t["lr_drop_step"], 0.1)
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "detr"), cfg)
    model.train()
    for step in range(1, t["steps"] + 1):
        imgs, targets = synthetic_detection(t["batch_size"], d["image_size"], d["n_classes"], d["max_objects"], g)
        imgs = imgs.to(device)
        targets = [{k: v.to(device) for k, v in tt.items()} for tt in targets]
        loss, parts = crit(model(imgs), targets)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip_max_norm"])
        opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "loss": loss.item(), **parts})
    model.eval()
    imgs, targets = synthetic_detection(32, d["image_size"], d["n_classes"], d["max_objects"], g)
    rec = recall_at_05(model, imgs.to(device), [{k: v.to(device) for k, v in tt.items()} for tt in targets])
    return logger.summary({"params": count_params(model), "final_loss": loss.item(), "recall_iou50": rec,
                           "is_paper_result": False if smoke else None})
