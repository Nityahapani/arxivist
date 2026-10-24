# Assumption and ambiguity report — arXiv:2108.12409

## Verified directly from the paper

- No position embeddings at the input.
- Bias −m·(i−j) added to query-key scores.
- For 8 heads, slopes 1/2^1 … 1/2^8.

## Explicit assumptions in this reproduction

1. **Embeddings:** Use standard tied embeddings instead of adaptive input / adaptive softmax. *Basis:* Simplifies the Baevski & Auli (2018) architecture; perplexities will not match exactly.
2. **Optimizer recipe:** Adam (0.9, 0.98), LR 1e-4 with warmup, clip 0.1, 286k updates. *Basis:* Approximation of the fairseq WikiText-103 recipe; verify against the paper/codebase.
3. **Non-power-of-two heads:** Use the interleaving rule from the reference implementation for head counts that are not powers of two. *Basis:* Paper describes the power-of-two case.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2108_12409/sir.json`) and in the README. None is presented as a recovered original run artifact.
