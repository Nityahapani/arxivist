"""Promptable-segmentation training loop with simulated interaction (Section 3, Appendix A)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import synthetic_instances
from .losses import mask_iou, sam_loss
from .model import SAM
from .prompts import corrective_point, noisy_box, sample_point
from .utils import RunLogger, count_params, get_device, set_seed


def build_model(m: dict) -> SAM:
    return SAM(m["img_size"], m["patch"], m["dim"], m["depth"], m["heads"], m["window_size"], m.get("global_attn"),
               m["embed_dim"], m["dec_depth"], m["dec_heads"], m["dec_mlp"])


def interactive_step(model, img_emb, gt_full, rounds, g):
    """One mask, `rounds` interactive iterations. Returns summed loss and final IoU."""
    S = model.img_size
    low_hw = model.prompt_encoder.mask_input_size
    gt_low = F.interpolate(gt_full[None, None].float(), low_hw, mode="nearest")[0]
    gt = gt_full.bool()
    if torch.rand(1, generator=g) < 0.5:
        pts, lbl, box = sample_point(gt, g)[None, None], torch.ones(1, 1, dtype=torch.long), None
    else:
        pts, lbl, box = None, None, noisy_box(gt, g)[None]
    mask_in, total, iou = None, 0.0, torch.tensor(0.0)
    for r in range(rounds):
        # multiple masks only for a single (ambiguous) point prompt; otherwise the 4th (single-mask) token
        multimask = r == 0 and box is None
        low, iou_pred = model.predict(img_emb, (pts, lbl) if pts is not None else None, box, mask_in, multimask)
        loss, best = sam_loss(low, iou_pred, gt_low)
        total = total + loss
        chosen = low[:, best[0]:best[0] + 1]
        iou = mask_iou(chosen.detach(), gt_low)[0, 0]
        mask_in = chosen.detach()
        pred_full = model.upscale(mask_in)[0, 0] > 0
        p, l = corrective_point(pred_full, gt, g)
        if p is not None and r < rounds - 1:
            pts = p[None, None] if pts is None else torch.cat([pts, p[None, None]], 1)
            lab = torch.tensor([[l]])
            lbl = lab if lbl is None else torch.cat([lbl, lab], 1)
    return total / rounds, iou


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    imgs, masks = synthetic_instances(d["n_train"], m["img_size"], d["max_objects"], g)
    model = build_model(m).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=(0.9, 0.999), weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / t["warmup"]))
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "sam"), cfg)
    for step in range(1, t["steps"] + 1):
        losses, ious = [], []
        for _ in range(t["batch_size"]):
            i = int(torch.randint(0, len(imgs), (1,), generator=g))
            if not masks[i]:
                continue
            emb = model.image_encoder(imgs[i:i + 1].to(device))
            gt = masks[i][int(torch.randint(0, len(masks[i]), (1,), generator=g))].to(device)
            l, iou = interactive_step(model, emb, gt, t["rounds"], g)
            losses.append(l); ious.append(iou.item())
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "loss": loss.item(), "mask_iou": sum(ious) / len(ious)})
    return logger.summary({"params": count_params(model), "final_loss": loss.item(), "final_mask_iou": sum(ious) / len(ious),
                           "is_paper_result": False if smoke else None})
