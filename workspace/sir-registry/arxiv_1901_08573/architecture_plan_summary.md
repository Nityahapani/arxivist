# Architecture / Reproduction Plan — TRADES (arXiv:1901.08573)

**Paper:** [Zhang et al. (2019), *Theoretically Principled Trade-off between Robustness and Accuracy*](https://arxiv.org/abs/1901.08573)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_1901_08573/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`yaodongyu/TRADES`](https://github.com/yaodongyu/TRADES) consulted only for training-schedule details. This repository is not the authors' official code.

## Core training step

```text
x ─► x' = x + 0.001·N(0,I) ─► 10 × [x' ← Π_ε(x' + 0.007·sign ∇ KL(f(x)||f(x')))]
loss = CE(f(x), y) + β · KL(f(x) || f(x'))
```

## Protocol

- CIFAR-10, WRN-34-10, SGD, batch 128, β ∈ {1, 6}; evaluate natural accuracy and PGD-20 robust accuracy.

## Reproduction assumptions and limits

- Train 76 epochs with LR 0.1 decayed x0.1 at epoch 75, batch 128, weight decay 2e-4.
- PGD-20 on cross-entropy, step 0.003, random uniform start in the eps-ball.
- No normalisation layer; inputs in [0, 1] so eps=0.031 is in pixel units.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
