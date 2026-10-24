"""DINOv2 pretraining step (Section 4) and k-NN evaluation of the teacher backbone.

loss = DINO([CLS], all student crops vs teacher global crops)
     + iBOT(masked patches of student global crops vs teacher)
     + 0.1 * KoLeo(student [CLS] of global crops)
Teacher: Sinkhorn-Knopp centring, temperature warmup 0.04 -> 0.07, EMA momentum cosine -> 1.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .augment import MultiCrop
from .losses import dino_loss, ibot_loss, koleo_loss, sinkhorn_knopp
from .masking import sample_masks
from .model import DinoV2
from .schedules import cosine, lr_at, teacher_temp_at
from .synth_images import synthetic_images
from .utils import RunLogger, count_params, get_device, set_seed
from .vit import DinoViT


def param_groups(student, weight_decay, layer_decay):
    """Layer-wise LR decay (rate `layer_decay` per block from the top); no WD on biases, norms, tokens, LayerScale."""
    depth = len(student["backbone"].blocks)
    groups = {}
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone.blocks."):
            layer = int(n.split(".")[2]) + 1
        elif n.startswith("backbone."):
            layer = 0
        else:
            layer = depth + 1
        no_wd = p.ndim == 1 or n.endswith(("cls_token", "mask_token", "pos_embed"))
        key = (layer, no_wd)
        if key not in groups:
            groups[key] = dict(params=[], lr_scale=layer_decay ** (depth + 1 - layer), wd_scale=0.0 if no_wd else 1.0)
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def knn_eval(backbone, xtr, ytr, xte, yte, k=20, T=0.07):
    backbone.eval()
    ftr, fte = F.normalize(backbone(xtr)[0], dim=-1), F.normalize(backbone(xte)[0], dim=-1)
    sim = fte @ ftr.t()
    w, idx = sim.topk(min(k, len(ftr)), dim=1)
    votes = torch.zeros(len(fte), int(ytr.max()) + 1).scatter_add_(1, ytr[idx], (w / T).exp())
    return (votes.argmax(1) == yte).float().mean().item()


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    backbone = DinoViT(m["global_size"], m["patch"], m["dim"], m["depth"], m["heads"], 4.0, m["drop_path"], m.get("swiglu", False))
    model = DinoV2(backbone, m["n_prototypes"], m["n_prototypes"], m["head_hidden"], m["head_bottleneck"]).to(device)
    groups = param_groups(model.student, t["wd_start"], t["layer_decay"])
    opt = torch.optim.AdamW(groups, lr=t["lr"], betas=(0.9, 0.999))
    aug = MultiCrop(m["global_size"], m["local_size"], t["n_local"], blur=t.get("blur", True))
    xtr, ytr = synthetic_images(d["n_train"], d["image_size"], d["n_classes"], g)
    xte, yte = synthetic_images(d["n_test"], d["image_size"], d["n_classes"], g)
    S = t["steps"]
    n_patches = (m["global_size"] // m["patch"]) ** 2
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "dinov2"), cfg)
    model.train()
    for step in range(S):
        lr, wd = lr_at(step, S, t["warmup"], t["lr"]), cosine(step, S, t["wd_start"], t["wd_end"])
        for gp in opt.param_groups:
            gp["lr"], gp["weight_decay"] = lr * gp["lr_scale"], wd * gp["wd_scale"]
        mom, ttemp = cosine(step, S, t["momentum_start"], 1.0), teacher_temp_at(step, t["teacher_temp_warmup"], t["teacher_temp_start"], t["teacher_temp_end"])
        x = xtr[torch.randint(0, len(xtr), (t["batch_size"],), generator=g)]
        globals_, locals_ = aug(x, g)
        globals_, locals_ = [v.to(device) for v in globals_], [v.to(device) for v in locals_]
        B = x.size(0)
        masks = torch.stack([sample_masks(B, n_patches, g, tuple(t["mask_ratio"]), t["mask_prob"]) for _ in range(2)]).to(device)
        with torch.no_grad():
            tg = [model.teacher["backbone"](v) for v in globals_]
            t_cls = [sinkhorn_knopp(model.teacher["dino_head"](c), ttemp) for c, _ in tg]
            t_patch = [sinkhorn_knopp(model.teacher["ibot_head"](p[mk]), ttemp) for (_, p), mk in zip(tg, masks)]
        sg = [model.student["backbone"](v, mk) for v, mk in zip(globals_, masks)]
        sl = [model.student["backbone"](v) for v in locals_]
        s_cls = [model.student["dino_head"](c) for c, _ in sg + sl]
        l_dino = dino_loss(s_cls, t_cls, t["student_temp"])
        l_ibot = sum(ibot_loss(model.student["ibot_head"](p), tp, mk, t["student_temp"]) for (_, p), tp, mk in zip(sg, t_patch, masks)) / 2
        l_koleo = sum(koleo_loss(c) for c, _ in sg) / 2
        loss = l_dino + t["ibot_weight"] * l_ibot + t["koleo_weight"] * l_koleo
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), t["clip_grad"])
        opt.step()
        model.update_teacher(mom)
        if (step + 1) % t["log_every"] == 0 or step + 1 == S:
            logger.log({"step": step + 1, "loss": loss.item(), "dino": l_dino.item(), "ibot": l_ibot.item(), "koleo": l_koleo.item(),
                        "lr": lr, "wd": wd, "momentum": mom, "teacher_temp": ttemp})
    import torch.nn.functional as Fn
    resize = lambda z: Fn.interpolate(z, size=m["global_size"], mode="bilinear", align_corners=False)
    acc = knn_eval(model.teacher["backbone"].cpu(), resize(xtr), ytr, resize(xte), yte)
    return logger.summary({"backbone_params": count_params(backbone), "final_loss": loss.item(), "knn_top1": acc,
                           "is_paper_result": False if smoke else None})
