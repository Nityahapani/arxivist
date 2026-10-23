# Architecture / Reproduction Plan — Grokking (arXiv:2201.02177)

**Paper:** [Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*](https://arxiv.org/abs/2201.02177)  
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2201_02177/README.md)  
**Evidence source:** paper v1, especially Sections 2–3 and Appendix A.1–A.5; authors' public [`openai/grok`](https://github.com/openai/grok) source used only to resolve implementation details. This repository is not the authors' official code.

## Core model

```text
atomic equation symbols [x, op, y, =]
        ↓
separate symbol embedding + sinusoidal positions
        ↓
2 × decoder block
  ├─ 4-head causal self-attention (width 128; key width 32)
  ├─ residual connection + LayerNorm
  ├─ 4×-width ReLU feed-forward layer
  └─ residual connection + LayerNorm
        ↓
hidden state at '=' → untied linear unembedding → answer-token logits
```

The paper reports about **4×10⁵ non-embedding parameters**. Loss and exact accuracy are computed for the answer symbol. The independent code follows the released authors' post-residual LayerNorm, sinusoidal position encoding, separate per-head Q/K/V projections, and 4× ReLU MLP. It uses a leading EOS token and answer-token-only loss to follow the paper prose; the public author code's shifted EOS-wrapped target slice also scores a trailing EOS, so this is disclosed as a small implementation difference.

## Dataset inventory (all Appendix A.1.1 operations)

All operands are separate atomic tokens. Modular operations use **p=97**; S5 has 120 permutations and 14,400 ordered pairs.

| Key | Operation |
|---|---|
| `mod_add` | x + y (mod 97) |
| `mod_subtract` | x − y (mod 97) |
| `mod_divide` | x / y (mod 97), y ≠ 0 |
| `mod_hybrid_div_sub` | x/y for odd y; x−y for even y (mod 97) |
| `mod_squares_sum` | x² + y² (mod 97) |
| `mod_quadratic` | x² + xy + y² (mod 97) |
| `mod_quadratic_plus_x` | x² + xy + y² + x (mod 97) |
| `mod_cubic_xy` | x³ + xy (mod 97) |
| `mod_cubic_xy2_y` | x³ + xy² + y (mod 97) |
| `s5_product` | x·y in S5 |
| `s5_conjugation` | x·y·x⁻¹ in S5 |
| `s5_xyx` | x·y·x in S5 |

For permutation arrays, the implementation follows the authors' data-code convention `(x*y)[i] = y[x[i]]`. The paper does not specify this encoding convention.

## Training protocol

- Default: AdamW, learning rate **1e-3**, weight decay **1**, β=(0.9, 0.98), 10-update linear warmup, constant learning rate, batch size `min(512, ceil(n_train/2))`, 100k updates.
- Figure 1 late-generalization run: Adam, no weight decay, 1M updates on modular division at 50% train data.
- Figure 1 S5 time-to-99% curve: Adam/no decay, 500k updates, seven seeds.
- Figure 2: three seeds; 100k updates; 12 operations plus nine S5 optimizer/regularization variants.
- Appendix A.4: k ∈ {0, 10, 100, 1000, 2000, 3000} training-label outliers; validation stays clean.
- Appendix A.5: S5 product, 70% data, 20 seeds; Keskar-style low-dimensional sharpness with a 10-dimensional random subspace, ε=1e-3, and 10 L-BFGS-B iterations.

## Reproduction assumptions and limits

The paper does not list every data fraction, name the operation used in Appendix A.4, or specify the t-SNE settings and the exact Figure 3 / Figure 7 checkpoints. `configs/experiment_suite.json` exposes those choices: 5-point data-fraction grids (denser near 25–32% for the S5 timing curve), modular addition for the outlier study, representative 50% embedding runs, and 70% / 100k-step / 20-seed sharpness runs. These are marked as assumptions in `sir.json` and in the paper-reproduction README. The results must not be described as exact recovery of unpublished author-run logs.

## Expected outputs

- Figure 1: modular-division learning curves and S5 product time-to-99% curve.
- Figure 2: nine-condition optimizer/regularization ablation and best validation accuracy for all 12 operations.
- Figure 3: t-SNE of learned unembedding weights for modular addition and S5.
- Figure 4: train/validation loss curves corresponding to Figure 1.
- Figure 5: generated operation truth table (CSV export).
- Figure 6: data efficiency under the six stated outlier counts.
- Figure 7: validation accuracy versus local sharpness, with Spearman correlation.
