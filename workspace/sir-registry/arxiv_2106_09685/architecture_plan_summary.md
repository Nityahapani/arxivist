# Architecture / Reproduction Plan — LoRA (arXiv:2106.09685)

**Paper:** [Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2106_09685/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`microsoft/LoRA`](https://github.com/microsoft/LoRA) (`loralib`) consulted only for implementation conventions. This repository is not the authors' official code.

## Core mechanism

```text
x ──► W0 (frozen) ───────────────┐
 └──► A (r×k) ──► B (d×r) ── ×α/r ┴─► h
```

At inference `W = W0 + (α/r)·BA` can be merged, so the model has exactly the original architecture.

## Protocol

- RoBERTa-base (125M) from Hugging Face; LoRA r=8, α=8 on `query`/`value` in all 12 layers.
- GLUE dev tasks; per-task batch size / epochs / LR in `configs/paper.json`; AdamW, linear schedule, 6% warmup.
- Median of 5 seeds per task; metrics: accuracy, MCC (CoLA), Pearson (STS-B).

## Expected outputs

- Per-task dev metrics and the GLUE average to compare with Table 2 (87.2).
- Trainable parameter count (~0.3M LoRA params + head).

## Reproduction assumptions and limits

- Initialise A from N(0, (1/r)^2) and B to zero; kaiming-uniform (loralib default) available via init='kaiming'.
- Use the RoBERTa-base per-task batch size / epochs / learning rate transcribed into configs/paper.json from the appendix table; contributors should verify each value against the paper.
- Train the task classifier head together with LoRA matrices; all other weights frozen.
- CPU smoke test uses a synthetic pretrain -> adapt task with a 2-layer, width-32 model; it validates mechanics only.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
