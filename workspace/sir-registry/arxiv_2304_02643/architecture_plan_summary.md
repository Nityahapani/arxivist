# Architecture / Reproduction Plan — Segment Anything (arXiv:2304.02643)

**Paper:** [Kirillov et al. (2023), *Segment Anything*](https://arxiv.org/abs/2304.02643)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2304_02643/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Architecture

```text
image ─ViT (windowed + 4 global)─▶ 64×64×256 ─┐
points/box ─Fourier PE + type emb─▶ tokens ───┤─ two-way transformer ×2 ─▶ mask tokens · upscaled emb ─▶ 3 masks + IoU
mask ─conv downscale─▶ dense (added) ─────────┘
```

## Reproduction assumptions and limits

- AdamW lr 8e-4, warmup 250, 90k iterations, batch 256, weight decay 0.1, step decay, drop path 0.4, layer decay 0.8 (configs/paper.json).
- Multiple masks only on the first round when no box is given; otherwise the single-mask token.
- No relative positional embeddings in windowed attention; random init instead of MAE pretraining.
- Text-prompt (CLIP) experiments are not implemented.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
