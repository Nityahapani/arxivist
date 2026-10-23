# Architecture / Reproduction Plan — Swin Transformer (arXiv:2103.14030)

**Paper:** [Liu et al. (2021), *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*](https://arxiv.org/abs/2103.14030)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2103_14030/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`microsoft/Swin-Transformer`](https://github.com/microsoft/Swin-Transformer) consulted only for the shifted-window mask construction. This repository is not the authors' official code.

## Stage structure (Swin-T)

```text
4×4 patch embed (C=96) → [W-MSA, SW-MSA]×1 → merge → ×1 (192) → merge → ×3 (384) → merge → ×1 (768) → GAP → FC
```

SW-MSA: roll by −M/2, window attention with region mask, roll back.

## Reproduction assumptions and limits

- RandAugment and Random Erasing are not implemented; Mixup/CutMix/label smoothing are.
- If a stage's resolution is <= window size, the window equals the full map and no shift is applied.
- Clip-norm 5.0 from the reference recipe is documented but not applied in the scaffold loop.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
