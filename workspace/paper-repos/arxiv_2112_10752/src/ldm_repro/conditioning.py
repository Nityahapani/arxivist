"""Domain-specific encoders tau_theta (Section 3.3): a class embedder and a small transformer text encoder
(the paper uses a BERT tokenizer + transformer for text-to-image on LAION)."""
from __future__ import annotations

import torch
from torch import nn


class ClassEmbedder(nn.Module):
    def __init__(self, n_classes, d):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, d)  # last index = null class for classifier-free guidance
        self.null = n_classes

    def forward(self, y):
        return self.emb(y).unsqueeze(1)  # [B, 1, d] context sequence


class TextEncoder(nn.Module):
    def __init__(self, vocab, d=512, layers=4, heads=8, max_len=77):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Parameter(torch.zeros(1, max_len, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        return self.ln(self.enc(self.tok(ids) + self.pos[:, : ids.size(1)]))
