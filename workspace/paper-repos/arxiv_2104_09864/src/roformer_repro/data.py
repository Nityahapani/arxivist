"""Synthetic data for masked language modelling whose tokens depend on relative position:
each sequence repeats a random motif with a random period, so predicting a masked token requires
attending to the token exactly one period away."""
from __future__ import annotations

import torch


def periodic_sequences(n, T, vocab, g, periods=(3, 8)):
    out = torch.empty(n, T, dtype=torch.long)
    for i in range(n):
        p = int(torch.randint(periods[0], periods[1] + 1, (1,), generator=g))
        motif = torch.randint(2, vocab, (p,), generator=g)  # ids 0/1 reserved for [PAD]/[MASK]
        out[i] = motif.repeat(T // p + 1)[:T]
    shift = torch.randint(0, T, (n,), generator=g)  # random phase -> absolute position uninformative
    return torch.stack([torch.roll(s, int(k)) for s, k in zip(out, shift)])


def mask_tokens(ids, g, mask_id=1, p=0.15):
    """BERT-style masking: 15% of tokens, of which 80% [MASK], 10% random, 10% unchanged."""
    labels = ids.clone()
    sel = torch.rand(ids.shape, generator=g) < p
    labels[~sel] = -100
    x = ids.clone()
    r = torch.rand(ids.shape, generator=g)
    x[sel & (r < 0.8)] = mask_id
    rnd = sel & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(2, int(ids.max()) + 1, (int(rnd.sum()),), generator=g)
    return x, labels
