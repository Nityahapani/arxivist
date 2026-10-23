# Architecture / Reproduction Plan — SimCLR (arXiv:2002.05709)

**Paper:** [Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations*](https://arxiv.org/abs/2002.05709)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2002_05709/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`google-research/simclr`](https://github.com/google-research/simclr) consulted only for augmentation conventions. This repository is not the authors' official code.

## Core model

```text
x ─► t(x), t'(x) ─► ResNet f ─► h ─► MLP g ─► z ─► NT-Xent(z_i, z_j)
```

## Protocol (CIFAR-10)

- ResNet-50 with 3×3 stem, no max-pool; batch 1,024; τ=0.5; colour strength 0.5; no blur; LARS; ~1,000 epochs.
- Linear evaluation on frozen features.

## Expected outputs

- Contrastive loss curve; linear-eval top-1 on CIFAR-10 test set.

## Reproduction assumptions and limits

- Run the CIFAR-10 variant (ResNet-50, CIFAR stem, no blur, batch 1024, tau 0.5, colour strength 0.5, ~1000 epochs) as the primary reproducible target.
- Linear evaluation uses Nesterov-momentum SGD on frozen, pre-extracted features without augmentation.
- Colour jitter is implemented in-tensor (brightness/contrast/saturation 0.8s, hue 0.2s, random order).
- LARS with trust coefficient 0.001, momentum 0.9; 1-D parameters (bias, BN) excluded from adaptation and weight decay.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
