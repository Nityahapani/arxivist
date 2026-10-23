"""The Prior-Data Fitted Network (Section 3-4, Fig. 1).

Each sample is a token: a linear encoding of its features, plus a label embedding for training samples.
Training tokens attend to all training tokens; test tokens attend only to training tokens (never to each
other), so predictions for a test point are independent of other test points and invariant to the order
of the training set (no positional encodings). Output: logits over up to `max_classes` classes.
"""
from __future__ import annotations

import torch
from torch import nn


def pfn_attention_mask(n_train: int, n_total: int, device=None) -> torch.Tensor:
    """Boolean mask [n_total, n_total], True = may attend. Every token attends to train tokens only."""
    m = torch.zeros(n_total, n_total, dtype=torch.bool, device=device)
    m[:, :n_train] = True
    return m


class TabPFN(nn.Module):
    def __init__(self, max_features=100, max_classes=10, d=512, n_layers=12, n_heads=4, ff=1024, dropout=0.0):
        super().__init__()
        self.x_enc = nn.Linear(max_features, d)
        self.y_enc = nn.Embedding(max_classes, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout, batch_first=True, norm_first=False, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.decoder = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, max_classes))

    def forward(self, x_train, y_train, x_test):
        """x_*: [B, n, F]; y_train: [B, n_train]. Returns logits [B, n_test, C]."""
        n_tr = x_train.size(1)
        tokens = torch.cat([self.x_enc(x_train) + self.y_enc(y_train), self.x_enc(x_test)], 1)
        mask = pfn_attention_mask(n_tr, tokens.size(1), tokens.device)
        h = self.encoder(tokens, mask=~mask)  # PyTorch: True = masked out
        return self.decoder(h[:, n_tr:])


@torch.no_grad()
def predict_proba(model, x_train, y_train, x_test, n_classes, n_ensemble=8, g=None, max_features=100):
    """Ensemble over random feature permutations and class relabelings (Section 4, 'ensembling')."""
    from .prior import normalize, pad_features

    X = torch.cat([x_train, x_test], 0).float()
    X = normalize(X)
    n_tr, d = len(x_train), X.size(1)
    probs = 0
    for _ in range(n_ensemble):
        fp = torch.randperm(d, generator=g)
        cp = torch.randperm(n_classes, generator=g)
        Xp = pad_features(X[:, fp], max_features).unsqueeze(0)
        logits = model(Xp[:, :n_tr], cp[y_train].unsqueeze(0), Xp[:, n_tr:])[0, :, :n_classes]
        probs = probs + logits.softmax(-1)[:, cp]  # undo the class permutation
    return probs / n_ensemble
