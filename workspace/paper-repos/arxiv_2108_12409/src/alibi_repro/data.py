"""Token streams: synthetic (smoke) and WikiText-103 (paper, optional HF datasets)."""
from __future__ import annotations

import torch


def synthetic_stream(n_tokens: int, vocab: int, g: torch.Generator, order: int = 2) -> torch.Tensor:
    """A random sparse order-k Markov chain: predictable from local context, any length."""
    table = torch.randint(0, vocab, (vocab,) * order, generator=torch.Generator().manual_seed(7))
    out = torch.randint(0, vocab, (n_tokens,), generator=g)
    noise = torch.rand(n_tokens, generator=g) < 0.1
    for t in range(order, n_tokens):
        if not noise[t]:
            out[t] = table[tuple(out[t - order:t].tolist())]
    return out


def wikitext103(split: str):  # pragma: no cover - requires network and `datasets`
    """Word-level WikiText-103 as a flat id tensor (vocabulary built from the train split)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-v1")
    vocab = {}
    for line in ds["train"]["text"]:
        for w in line.split() + ["<eos>"]:
            vocab.setdefault(w, len(vocab))
    ids = [vocab.get(w, vocab.get("<unk>", 0)) for line in ds[split]["text"] for w in line.split() + ["<eos>"]]
    return torch.tensor(ids), len(vocab)


def batches(stream: torch.Tensor, seq_len: int, batch_size: int, g: torch.Generator):
    while True:
        starts = torch.randint(0, len(stream) - seq_len - 1, (batch_size,), generator=g)
        x = torch.stack([stream[s:s + seq_len + 1] for s in starts])
        yield x[:, :-1], x[:, 1:]
