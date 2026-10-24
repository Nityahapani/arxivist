# Architecture / Reproduction Plan — Latent Diffusion (arXiv:2112.10752)

**Paper:** [Rombach et al. (2022), *High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2112_10752/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Two stages

```text
stage 1:  x ─E─▶ z ─D─▶ x̂        (L1 + KL + PatchGAN)
stage 2:  z_t = √ᾱ z + √(1−ᾱ) ε;  min ‖ε − ε_θ(z_t, t, τ(y))‖²
sample:   DDIM in latent space (+ CFG) ─D─▶ image
```

## Reproduction assumptions and limits

- LPIPS is not implemented (needs pretrained VGG weights); first stage uses L1 + KL + PatchGAN.
- 'Scaled linear' β schedule 0.00085 → 0.012 with T = 1000 (as in the released LDM configs).
- Channel multipliers, attention resolutions and iteration counts in configs/paper.json are best-recalled values.
- Latents are rescaled by 1/std estimated on the training set before diffusion (KL models).

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
