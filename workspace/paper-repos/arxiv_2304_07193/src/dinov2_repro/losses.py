"""DINOv2 objectives (Section 4).

* Image-level DINO loss: cross-entropy between teacher and student prototype distributions of [CLS]
  tokens of different crops; the teacher is centred with Sinkhorn-Knopp (3 iterations, from SwAV) and
  sharpened with a low temperature.
* Patch-level iBOT loss: the student sees masked global crops; cross-entropy between teacher and student
  patch prototypes at masked positions.
* KoLeo regulariser (Sablayrolles et al.): L = -1/n sum_i log(d_{n,i}), d_{n,i} = min_{j != i} ||x_i - x_j||
  on L2-normalised student [CLS] features within a batch (weight 0.1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(teacher_logits: torch.Tensor, temp: float, n_iters: int = 3) -> torch.Tensor:
    """teacher_logits [B, K] -> targets [B, K] (rows sum to 1; prototypes used equally across the batch)."""
    Q = torch.exp((teacher_logits.float() - teacher_logits.float().max()) / temp).t()  # [K, B]
    K, B = Q.shape
    Q /= Q.sum()
    for _ in range(n_iters):
        Q /= Q.sum(1, keepdim=True); Q /= K
        Q /= Q.sum(0, keepdim=True); Q /= B
    return (Q * B).t()


def dino_loss(student_logits: list, teacher_probs: list, student_temp: float = 0.1) -> torch.Tensor:
    """student_logits: list over student crops of [B, K]; teacher_probs: list over the 2 global crops.
    Pairs with the same global crop index are skipped."""
    total, n = 0.0, 0
    for ti, t in enumerate(teacher_probs):
        for si, s in enumerate(student_logits):
            if si == ti:
                continue
            total = total + torch.sum(-t * F.log_softmax(s / student_temp, -1), -1).mean()
            n += 1
    return total / n


def ibot_loss(student_patch_logits, teacher_patch_probs, masks, student_temp: float = 0.1) -> torch.Tensor:
    """student_patch_logits [B, N, K]; teacher_patch_probs [M, K] for masked positions (row order of masks.nonzero())."""
    s = student_patch_logits[masks]
    per = torch.sum(-teacher_patch_probs * F.log_softmax(s / student_temp, -1), -1)
    # weight each image equally regardless of how many patches it masks
    w = (1.0 / masks.sum(1).clamp_min(1)).unsqueeze(1).expand_as(masks)[masks]
    return (per * w).sum() / masks.size(0)


def koleo_loss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    d = torch.cdist(x, x)
    d = d + torch.eye(len(x), device=x.device) * 1e9
    return -torch.log(d.min(1).values + eps).mean()
