"""Synthetic controlled-generation task (analogue of the IMDb sentiment experiment, Section 6.1).

The ground-truth reward of a response is the fraction of its tokens drawn from a "positive" subset of
the vocabulary. Preference labels are sampled from the Bradley-Terry model p(y1 > y2) = sigma(r1 - r2)
(Eq. 1) after scaling the reward by `reward_scale`.
"""
from __future__ import annotations

import torch


def true_reward(responses: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    return torch.isin(responses, positive).float().mean(-1)


def random_prompts(n, prompt_len, vocab, g):
    return torch.randint(0, vocab, (n, prompt_len), generator=g)


def bradley_terry_labels(r1, r2, reward_scale, g):
    """Returns a bool tensor: True where y1 is preferred."""
    p = torch.sigmoid(reward_scale * (r1 - r2))
    return torch.rand(p.shape, generator=g) < p


def build_pairs(prompts, y1, y2, positive, reward_scale, g):
    r1, r2 = true_reward(y1, positive), true_reward(y2, positive)
    first = bradley_terry_labels(r1, r2, reward_scale, g).unsqueeze(-1)
    chosen = torch.where(first, y1, y2)
    rejected = torch.where(first, y2, y1)
    return torch.cat([prompts, chosen], 1), torch.cat([prompts, rejected], 1)


def response_mask(batch: torch.Tensor, prompt_len: int) -> torch.Tensor:
    m = torch.zeros_like(batch, dtype=torch.bool)
    m[:, prompt_len:] = True
    return m
