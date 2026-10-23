# Architecture / Reproduction Plan — U-Net (arXiv:1505.04597)

**Paper:** [Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_1505_04597/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Architecture

```text
572 → [conv3 conv3 pool]×4 → conv3 conv3 (1024, dropout) → [upconv2 · crop-concat · conv3 conv3]×4 → conv1 → 388
```

## Reproduction assumptions and limits

- LR 0.01 and 10k iterations.
- The border term is applied to background pixels between cells.
- Coarse 3x3 displacement field upsampled with cubic spline zoom (scipy).
- Synthetic touching disks replace EM/cell data for CI.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
