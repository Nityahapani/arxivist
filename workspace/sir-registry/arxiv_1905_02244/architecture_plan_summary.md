# Architecture / Reproduction Plan — MobileNetV3 (arXiv:1905.02244)

**Paper:** [Howard et al. (2019), *Searching for MobileNetV3*](https://arxiv.org/abs/1905.02244)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_1905_02244/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Large 1.0

```text
conv3×3/2 (16, h-swish) → 15 bnecks (Table 1) → conv1×1 960 (h-swish) → GAP → FC 1280 (h-swish, no BN) → FC 1000
```

## Reproduction assumptions and limits

- Interpret 'learning rate decay rate of 0.01 every 3 epochs' as multiplying LR by 0.97 every 3 epochs.
- Interpret 'dropout of 0.8' as keep-probability 0.8 (dropout rate 0.2) before the classifier.
- Use eps=0.0316 (≈ sqrt(1e-3)) to mimic TensorFlow's epsilon placement.
- ~600 epochs (187,500 steps at batch 4096).

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
