"""Perplexity evaluation: non-overlapping and sliding-window (Section 2 / Appendix B)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def nonoverlapping_ppl(model, stream, L, device="cpu", max_windows=None):
    model.eval()
    nll, n = 0.0, 0
    for k, s in enumerate(range(0, len(stream) - L - 1, L)):
        if max_windows and k >= max_windows:
            break
        x = stream[s:s + L + 1].to(device)
        logits = model(x[None, :-1])[0]
        nll += F.cross_entropy(logits, x[1:], reduction="sum").item()
        n += L
    return math.exp(nll / max(1, n))


@torch.no_grad()
def sliding_ppl(model, stream, L, stride, device="cpu", max_windows=None):
    """Each window of length L scores only its last `stride` tokens (stride=1 is the paper's
    sliding-window evaluation; larger strides trade accuracy for speed)."""
    model.eval()
    nll, n = 0.0, 0
    for k, s in enumerate(range(0, len(stream) - L - 1, stride)):
        if max_windows and k >= max_windows:
            break
        x = stream[s:s + L + 1].to(device)
        logits = model(x[None, :-1])[0, -stride:]
        nll += F.cross_entropy(logits, x[-stride:], reduction="sum").item()
        n += stride
    return math.exp(nll / max(1, n))
