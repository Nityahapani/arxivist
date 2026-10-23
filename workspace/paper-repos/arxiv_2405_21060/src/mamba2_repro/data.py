"""Multi-Query Associative Recall (MQAR) synthetic task (Section 9.1, following Zoology / Arora et al.).

Each sequence lists `n_kv` key-value pairs followed by queries of some keys; the model must output the
associated value after each queried key. Keys and values use disjoint vocab halves.
"""
from __future__ import annotations

import torch


def mqar_batch(batch, n_kv, n_queries, vocab, g):
    half = vocab // 2
    seq_len = 2 * n_kv + 2 * n_queries
    x = torch.zeros(batch, seq_len, dtype=torch.long)
    y = torch.full((batch, seq_len), -100, dtype=torch.long)
    for i in range(batch):
        keys = torch.randperm(half - 1, generator=g)[:n_kv] + 1
        vals = torch.randint(half, vocab, (n_kv,), generator=g)
        x[i, 0:2 * n_kv:2], x[i, 1:2 * n_kv:2] = keys, vals
        q = torch.randint(0, n_kv, (n_queries,), generator=g)
        pos = 2 * n_kv + 2 * torch.arange(n_queries)
        x[i, pos], x[i, pos + 1] = keys[q], vals[q]
        y[i, pos] = vals[q]  # predict the value right after seeing the query key
    return x, y
