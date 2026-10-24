"""Data for LoRA experiments.

* Synthetic tasks (CPU smoke test): a "pretraining" task and a distinct downstream task over the
  same vocabulary, so adaptation of a frozen pretrained model can be exercised end to end.
* GLUE (paper Table 2) through Hugging Face ``datasets``/``transformers`` (optional dependency).
"""
from __future__ import annotations

import torch

CLS = 0


def synthetic_task(n: int, seq_len: int, vocab: int, task: str, generator: torch.Generator):
    """Returns (ids [n, seq_len], labels [n]).

    ``majority``: label = whether token 1 occurs more often than token 2 (pretraining task).
    ``first_half``: label = whether the sum of the first half exceeds the second half (downstream).
    """
    ids = torch.randint(1, vocab, (n, seq_len), generator=generator)
    ids[:, 0] = CLS
    body = ids[:, 1:]
    if task == "majority":
        y = ((body == 1).sum(1) > (body == 2).sum(1)).long()
    elif task == "first_half":
        h = body.size(1) // 2
        y = (body[:, :h].sum(1) > body[:, h:].sum(1)).long()
    else:
        raise ValueError(task)
    return ids, y


GLUE_KEYS = {
    "mnli": ("premise", "hypothesis", "validation_matched"),
    "sst2": ("sentence", None, "validation"),
    "mrpc": ("sentence1", "sentence2", "validation"),
    "cola": ("sentence", None, "validation"),
    "qnli": ("question", "sentence", "validation"),
    "qqp": ("question1", "question2", "validation"),
    "rte": ("sentence1", "sentence2", "validation"),
    "stsb": ("sentence1", "sentence2", "validation"),
}


def load_glue(task: str, tokenizer, max_length: int = 512):  # pragma: no cover - needs network
    from datasets import load_dataset

    k1, k2, val = GLUE_KEYS[task]
    ds = load_dataset("glue", task)

    def enc(batch):
        args = (batch[k1],) if k2 is None else (batch[k1], batch[k2])
        return tokenizer(*args, truncation=True, max_length=max_length)

    ds = ds.map(enc, batched=True)
    return ds["train"], ds[val]
