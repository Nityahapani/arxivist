# Architecture / Reproduction Plan — ConvNeXt (arXiv:2201.03545)

**Paper:** [Liu et al. (2022), *A ConvNet for the 2020s*](https://arxiv.org/abs/2201.03545)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2201_03545/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`facebookresearch/ConvNeXt`](https://github.com/facebookresearch/ConvNeXt) consulted only for initialisation details. This repository is not the authors' official code.

## Block

```text
x ─► DWConv 7×7 ─► LN ─► Linear 4C ─► GELU ─► Linear C ─► γ· ─► DropPath ─► + x
```

## Network (ConvNeXt-T)

stem 4×4/4 → [3 blocks @96] → LN+2×2/2 → [3 @192] → … → [9 @384] → … → [3 @768] → GAP → LN → FC

## Reproduction assumptions and limits

- RandAugment and Random Erasing are not implemented in the scaffold; Mixup/CutMix/label smoothing are.
- EMA of weights (decay 0.9999) is optional and evaluated separately.
- An ImageNet loader must be supplied.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
