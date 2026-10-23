# Architecture / Reproduction Plan — DETR (arXiv:2005.12872)

**Paper:** [Carion et al. (2020), *End-to-End Object Detection with Transformers*](https://arxiv.org/abs/2005.12872)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2005_12872/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Pipeline

```text
image → ResNet-50 → 1×1 conv → +sine pos → encoder ×6 → decoder ×6 (100 queries) → class / box heads
loss: Hungarian match → CE(no-object 0.1) + 5·L1 + 2·GIoU, at every decoder layer
```

## Reproduction assumptions and limits

- Backbone is randomly initialised here; paper uses ImageNet-pretrained ResNet.
- Synthetic coloured rectangles in smoke runs; COCO loader and augmentations not included.
- Epochs converted to iterations with ~1848 iterations/epoch at batch 64.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
