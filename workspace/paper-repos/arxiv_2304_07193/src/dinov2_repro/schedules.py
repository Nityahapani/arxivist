"""Cosine schedules used by DINOv2: LR (warmup + cosine), weight decay 0.04 -> 0.2, teacher momentum
-> 1, and linear warmup of the teacher temperature 0.04 -> 0.07."""
import math


def cosine(step, total, start, end):
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * min(step, total) / max(1, total)))


def lr_at(step, total, warmup, base, final=1e-6):
    if step < warmup:
        return base * (step + 1) / warmup
    return cosine(step - warmup, total - warmup, base, final)


def teacher_temp_at(step, warmup, start=0.04, end=0.07):
    return end if step >= warmup else start + (end - start) * step / max(1, warmup)
