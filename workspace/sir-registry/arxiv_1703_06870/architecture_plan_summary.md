# Architecture / Reproduction Plan — Mask R-CNN (arXiv:1703.06870)

**Paper:** [He et al. (2017), *Mask R-CNN*](https://arxiv.org/abs/1703.06870)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_1703_06870/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Per-RoI branches

```text
P2–P5 ──RoIAlign 7×7──▶ FC1024 ×2 ──▶ class (K+1), box (4K)
      ──RoIAlign 14×14─▶ conv3×3 ×4 ─▶ deconv ─▶ K × 28×28 masks (sigmoid, GT class only)
```

## Reproduction assumptions and limits

- Default aligned=True (half-pixel offset, later Detectron2 convention); aligned=False reproduces original Detectron.
- Random init in this repo; paper uses ImageNet-pretrained backbones.
- FPN-style RPN (anchor sizes 32–512 on P2–P6, 256 anchors/image, NMS 0.7, 2000/1000 proposals).
- Linear warmup over 500 iterations in paper config (Detectron convention, not in paper).

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
