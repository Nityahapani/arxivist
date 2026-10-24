# Architecture / Reproduction Plan — MAE (arXiv:2111.06377)

**Paper:** [He et al. (2021), *Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2111_06377/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`facebookresearch/mae`](https://github.com/facebookresearch/mae) consulted only for initialisation and position-embedding conventions. This repository is not the authors' official code.

## Core model

```text
image ─► patches ─► +pos ─► random keep 25% ─► ViT encoder ─► Linear ─┐
                                   mask tokens (75%) ─────────────────┴► unshuffle + pos ─► 8-block decoder ─► pixels
loss = MSE(pred, norm_pix(target)) on masked patches
```

## Protocol

- ImageNet-1K pretraining (800 epochs ViT-L), then fine-tuning per Appendix A.2 (to be added).

## Reproduction assumptions and limits

- Pretraining and linear probe are implemented; full fine-tuning recipe (layer-wise LR decay, RandAug, mixup/cutmix, drop path) is not part of this scaffold.
- Probe on the mean of encoder patch tokens with a feature-standardisation step (stand-in for BatchNorm without affine).
- An ImageNet data loader must be supplied by the contributor (`data.name=imagefolder`).

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
