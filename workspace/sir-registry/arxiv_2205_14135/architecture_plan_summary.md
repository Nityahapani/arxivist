# Architecture / Reproduction Plan — FlashAttention (arXiv:2205.14135)

**Paper:** [Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2205_14135/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Forward tiling

```text
for each Q block i:      m=-inf, l=0, acc=0
  for each K,V block j:  S=Q_i K_jᵀ;  m'=max(m,rowmax S);  P=exp(S−m')
                         l = e^{m−m'} l + rowsum P;  acc = e^{m−m'} acc + P V_j
  O_i = acc / l;  L_i = m + log l
```

## Reproduction assumptions and limits

- Pure-PyTorch tiled implementation for correctness and IO accounting; no CUDA kernel.
- Attention dropout (regenerated from the RNG state in the backward pass) is not implemented.
- Element counts per (batch, head) for Algorithm 0 and Algorithm 1 with the Algorithm 2 loop order.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
