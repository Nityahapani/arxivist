# Architecture / Reproduction Plan — ALiBi (arXiv:2108.12409)

**Paper:** [Press et al. (2021), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*](https://arxiv.org/abs/2108.12409)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2108_12409/README.md)
**Evidence source:** the arXiv paper, with the authors' public [`ofirpress/attention_with_linear_biases`](https://github.com/ofirpress/attention_with_linear_biases) consulted only for the non-power-of-two slope rule. This repository is not the authors' official code.

## Attention

```text
scores_h = q K^T / sqrt(d_h) + causal_mask - m_h · (i - j)
```

## Protocol

- Train ALiBi / sinusoidal / rotary at L_train; evaluate perplexity at L_valid ≥ L_train on WikiText-103 validation.

## Reproduction assumptions and limits

- Use standard tied embeddings instead of adaptive input / adaptive softmax.
- Adam (0.9, 0.98), LR 1e-4 with warmup, clip 0.1, 286k updates.
- Use the interleaving rule from the reference implementation for head counts that are not powers of two.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
