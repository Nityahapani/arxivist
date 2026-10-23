# Architecture / Reproduction Plan — EfficientNet (arXiv:1905.11946)

**Paper:** [Tan & Le (2019), *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*](https://arxiv.org/abs/1905.11946)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_1905_11946/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Compound scaling

```text
depth  ×α^φ   (α=1.2)
width  ×β^φ   (β=1.1)
res    ×γ^φ   (γ=1.15)      α·β²·γ² ≈ 2  →  FLOPs ×≈2^φ
```

## Protocol

- Train B0 on ImageNet with the Section 5.2 recipe; optionally scale to B1–B7.

## Reproduction assumptions and limits

- Batch 4096 and 350 epochs from the authors' public TPU recipe.
- Label smoothing 0.1 (public recipe).
- AutoAugment and weight EMA are not implemented in the scaffold.
- SE squeezes to 0.25 x block input channels.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
