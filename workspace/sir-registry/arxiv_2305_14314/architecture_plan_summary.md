# Architecture / Reproduction Plan — QLoRA (arXiv:2305.14314)

**Paper:** [Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2305_14314/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## QLoRA layer

```text
W (bf16) ──NF4, block 64──▶ codes + absmax ──DQ 8-bit, block 256──▶ c2, c1
forward:  y = x · dequant(W) + (α/r) · x Aᵀ Bᵀ      (only A, B trainable)
```

## Reproduction assumptions and limits

- Asymmetric quantile construction with offset 0.9677083 (8 positive, 7 negative levels plus exact zero), matching the published NF4 table.
- Second-level constants are 8-bit symmetric integers per 256-block (centred by the mean).
- Not implemented (CPU reference); standard Adam is used.
- A tiny GPT pretrained in-script stands in for LLaMA.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
