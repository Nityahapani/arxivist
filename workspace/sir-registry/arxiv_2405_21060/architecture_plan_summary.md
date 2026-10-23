# Architecture / Reproduction Plan — Mamba-2 (arXiv:2405.21060)

**Paper:** [Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2405_21060/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## SSD chunked algorithm

```text
1. diagonal blocks:  Y_diag = (L ∘ C Bᵀ) X   within each chunk
2. chunk states:     s_c = Σ decay · B xᵀ
3. state passing:    recurrence over chunks
4. off-diagonal:     Y_off = C s_{c-1} · decay
```

## Reproduction assumptions and limits

- Only the 2–8× core-layer speed-up is quoted; read MQAR (Fig. 8), scaling (Fig. 9) and zero-shot (Table 1) targets from the paper.
- Pure PyTorch einsum implementation of Listing 1; no Triton kernels.
- A ~ -U[1,16], dt in [1e-3, 0.1] log-uniform, D = 1 (reference implementation defaults).
- AdamW (0.9, 0.95), wd 0.1, warmup-cosine; LR is an assumption.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
