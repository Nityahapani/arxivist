# Architecture / Reproduction Plan — BYOL (arXiv:2006.07733)

**Paper:** [Grill et al. (2020), *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*](https://arxiv.org/abs/2006.07733)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2006_07733/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Algorithm

```text
v, v' = t(x), t'(x)
loss  = ‖norm(q(g(f(v)))) − norm(g_ξ(f_ξ(v')))‖² + symmetric term
θ ← LARS(θ, ∇θ loss);   ξ ← τ ξ + (1 − τ) θ
```

## Reproduction assumptions and limits

- BatchNorm running statistics of the target network are copied from the online network.
- Short SGD linear probe on frozen features in smoke runs.
- Not included; synthetic data and CIFAR-10 only.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
