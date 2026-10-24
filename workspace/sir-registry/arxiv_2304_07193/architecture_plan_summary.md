# Architecture / Reproduction Plan — DINOv2 (arXiv:2304.07193)

**Paper:** [Oquab et al. (2023), *DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2304_07193/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Objective

```text
teacher(global crops) ─Sinkhorn─▶ targets
student(masked global + local crops) ─▶ DINO([CLS]) + iBOT(masked patches) + 0.1·KoLeo
teacher ← EMA(student)
```

## Reproduction assumptions and limits

- LR, warmup, prototype counts, head sizes and teacher momentum start in configs/paper.json are best-recalled values; verify against the paper's appendix.
- Random (not block-wise) patch masking with ratio U[0.1, 0.5] on 50% of global crops.
- FlashAttention/xFormers, sequence packing, FSDP and the high-resolution phase are not implemented.
- Synthetic images replace LVD-142M; the curation pipeline is not implemented.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
