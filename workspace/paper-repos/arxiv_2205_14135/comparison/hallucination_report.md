# Assumption and ambiguity report — arXiv:2205.14135

## Verified directly from the paper

- Algorithm 1 online softmax recurrences.
- Backward recomputation with D_i = rowsum(dO ∘ O).
- Block size formula.
- Theorem 2 asymptotics.

## Explicit assumptions in this reproduction

1. **Reference kernel:** Pure-PyTorch tiled implementation for correctness and IO accounting; no CUDA kernel. *Basis:* Speed claims need the CUDA kernel.
2. **Dropout:** Attention dropout (regenerated from the RNG state in the backward pass) is not implemented. *Basis:* Keeps the reference exact and deterministic.
3. **HBM model:** Element counts per (batch, head) for Algorithm 0 and Algorithm 1 with the Algorithm 2 loop order. *Basis:* Constants differ from the asymptotic statement.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2205_14135/sir.json`) and in the README. None is presented as a recovered original run artifact.
