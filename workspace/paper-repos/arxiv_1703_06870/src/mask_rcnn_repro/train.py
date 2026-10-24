"""Mask R-CNN training (Section 3.1): SGD momentum 0.9, weight decay 1e-4, LR 0.02 for batch 16,
step decay x0.1, multi-task loss; evaluation reports mask IoU of matched detections on synthetic data."""
from __future__ import annotations

from pathlib import Path

import torch

from .box_ops import box_iou
from .data import synthetic_instances
from .model import MaskRCNN
from .roi_heads import paste_masks
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def mask_iou_eval(model, imgs, targets):
    model.eval()
    res = model(imgs)
    ious = []
    for r, t in zip(res, targets):
        if len(r["boxes"]) == 0:
            ious += [0.0] * len(t["boxes"]); continue
        pm = paste_masks(r["masks"], r["boxes"], *imgs.shape[-2:])
        match = box_iou(t["boxes"], r["boxes"]).argmax(1)
        for gi, pi in enumerate(match.tolist()):
            inter = (pm[pi] & t["masks"][gi]).sum().item(); union = (pm[pi] | t["masks"][gi]).sum().item()
            ious.append(inter / max(1, union))
    return sum(ious) / max(1, len(ious))


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    model = MaskRCNN(d["n_classes"], m["arch"], m["base_width"], m["fpn_dim"], tuple(m["anchor_sizes"]), m["canonical"],
                     m["rpn_pre_nms"], m["rpn_post_nms"], m["rois_per_image"], m["fc"], m["mask_dim"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = t["base_lr"] * t["batch_size"] / 16
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, t["warmup"])) * (0.1 ** sum(s >= x for x in t["lr_steps"])))
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "mask_rcnn"), cfg)
    model.train()
    for step in range(1, t["steps"] + 1):
        imgs, targets = synthetic_instances(t["batch_size"], d["image_size"], d["max_objects"], g, d["n_classes"])
        losses = model(imgs.to(device), targets, g)
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "loss": loss.item(), **{k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}})
    imgs, targets = synthetic_instances(8, d["image_size"], d["max_objects"], g, d["n_classes"])
    miou = mask_iou_eval(model, imgs.to(device), targets)
    return logger.summary({"params": count_params(model), "final_loss": loss.item(), "mask_iou": miou, "is_paper_result": False if smoke else None})
