# Architecture / Reproduction Plan — RoFormer (arXiv:2104.09864)

**Paper:** [Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2104_09864/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## RoPE

```text
q_m ← R(mθ) q_m,   k_n ← R(nθ) k_n   ⇒   q_mᵀ k_n = qᵀ R((n−m)θ) k
```

## Reproduction assumptions and limits

- No numeric targets are hard-coded; read targets from Table 2 (GLUE), Table 5 (CAIL2019-SCM) and Figures 3–4 (pretraining curves).
- Interleaved pairs (x_{2i-1}, x_{2i}) as in Eq. 15.
- phi = elu + 1 for both queries and keys.
- AdamW, LR 1e-4, 10k warmup in configs/paper.json.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
