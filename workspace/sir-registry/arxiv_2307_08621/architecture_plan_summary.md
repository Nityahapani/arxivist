# Architecture / Reproduction Plan — RetNet (arXiv:2307.08621)

**Paper:** [Sun et al. (2023), *Retentive Network: A Successor to Transformer for Large Language Models*](https://arxiv.org/abs/2307.08621)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2307_08621/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Three forms

```text
parallel:  (Q Kᵀ ⊙ D) V
recurrent: S ← γS + kᵀv;  o = qS
chunkwise: inner-chunk parallel + (Q R_{i-1}) ⊙ ξ;  R_i = Kᵀ(V ⊙ ζ) + γᴮ R_{i-1}
```

## Reproduction assumptions and limits

- The paper's three retention normalisations (QK^T/sqrt(d), decay-matrix normalisation, score normalisation) are reduced to key scaling d^-1/2; GroupNorm's scale invariance makes the others numerically, not functionally, relevant.
- xPos-style Θ is implemented as RoPE without the xPos length-extrapolation scaling.
- AdamW (0.9, 0.98), wd 0.05, LR 6e-4, warmup 375, 25k steps for 1.3B.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
