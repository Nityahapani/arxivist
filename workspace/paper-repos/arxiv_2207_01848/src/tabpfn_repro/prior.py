"""Synthetic dataset priors (Hollmann et al., 2022, Section 4 and Appendix C).

* BNN prior: random MLP with random width/depth, activation, weight scale and dropout-style sparsity;
  inputs x ~ noise, target = discretised output of the network.
* SCM prior: a random MLP is interpreted as a structural causal model; features and the target are
  read from random intermediate nodes (so features can be causes, effects or unrelated), with Gaussian
  noise added at every node.
Targets are converted to classes by thresholding at random quantiles. Each call returns one dataset.
"""
from __future__ import annotations

import math

import torch

ACTS = [torch.tanh, torch.relu, lambda x: x, torch.sin, lambda x: torch.abs(x), lambda x: torch.sigmoid(x) - 0.5]


def _rand_int(lo, hi, g):
    return int(torch.randint(lo, hi + 1, (1,), generator=g))


def _log_uniform(lo, hi, g):
    return math.exp(torch.empty(1).uniform_(math.log(lo), math.log(hi), generator=g).item())


def _classes_from_target(y, n_classes, g):
    if n_classes == 1:
        return torch.zeros_like(y, dtype=torch.long)
    qs = torch.sort(torch.rand(n_classes - 1, generator=g)).values
    thresholds = torch.quantile(y, qs)
    labels = torch.bucketize(y.contiguous(), thresholds)
    perm = torch.randperm(n_classes, generator=g)  # random class order
    return perm[labels]


def sample_mlp_dataset(n, n_features, n_classes, g, scm=True):
    """Draw one dataset from the BNN (scm=False) or SCM (scm=True) prior."""
    depth, width = _rand_int(2, 5, g), _rand_int(max(8, n_features + 1), 64, g)
    act = ACTS[_rand_int(0, len(ACTS) - 1, g)]
    w_scale, noise = _log_uniform(0.5, 3.0, g), _log_uniform(1e-3, 0.3, g)
    in_dim = width if scm else n_features
    h = torch.randn(n, in_dim, generator=g) * (_log_uniform(0.1, 1.0, g) if scm else 1.0)
    nodes = [h]
    for _ in range(depth):
        W = torch.randn(h.size(1), width, generator=g) * w_scale / math.sqrt(h.size(1))
        W = W * (torch.rand(W.shape, generator=g) > _log_uniform(0.01, 0.9, g) * 0.5)  # sparse connections
        h = act(h @ W) + noise * torch.randn(n, width, generator=g)
        nodes.append(h)
    if scm:
        allnodes = torch.cat(nodes[1:], 1)
        idx = torch.randperm(allnodes.size(1), generator=g)[: n_features + 1]
        X, y = allnodes[:, idx[:-1]], allnodes[:, idx[-1]]
    else:
        X, y = nodes[0], h[:, _rand_int(0, width - 1, g)]
    return X, _classes_from_target(y, n_classes, g)


def sample_prior_batch(batch, n, max_features, max_classes, g, p_scm=0.5):
    """Returns X [batch, n, max_features] (zero padded, scaled by max_features / d), y [batch, n], n_features, n_classes."""
    Xs, ys = [], []
    d = _rand_int(1, max_features, g)
    c = _rand_int(2, max_classes, g)
    for _ in range(batch):
        X, y = sample_mlp_dataset(n, d, c, g, scm=bool(torch.rand(1, generator=g) < p_scm))
        Xs.append(X); ys.append(y)
    X = torch.stack(Xs)
    return pad_features(normalize(X), max_features), torch.stack(ys), d, c


def normalize(X, eps=1e-6):
    mean, std = X.mean(-2, keepdim=True), X.std(-2, keepdim=True)
    return ((X - mean) / (std + eps)).clamp(-100, 100)


def pad_features(X, max_features):
    """Zero-pad to `max_features` and rescale by max_features / d so the input magnitude is constant (App. A)."""
    d = X.size(-1)
    X = X * (max_features / d)
    return torch.cat([X, X.new_zeros(*X.shape[:-1], max_features - d)], -1)
