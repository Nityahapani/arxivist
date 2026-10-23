# Assumption and ambiguity report — arXiv:2405.21060

## Verified directly from the paper

- SSD recurrence and semiseparable mask.
- Chunked algorithm steps of Listing 1.
- Mamba-2 block layout of Fig. 6.

## Explicit assumptions in this reproduction

1. **Reported numbers:** Only the 2–8× core-layer speed-up is quoted; read MQAR (Fig. 8), scaling (Fig. 9) and zero-shot (Table 1) targets from the paper. *Basis:* Avoid citing uncertain numbers.
2. **Kernels:** Pure PyTorch einsum implementation of Listing 1; no Triton kernels. *Basis:* CPU-testable reference.
3. **Initialisation:** A ~ -U[1,16], dt in [1e-3, 0.1] log-uniform, D = 1 (reference implementation defaults). *Basis:* Not fully specified in text.
4. **Training hyperparameters:** AdamW (0.9, 0.95), wd 0.1, warmup-cosine; LR is an assumption. *Basis:* Paper follows GPT-3/Mamba recipes.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2405_21060/sir.json`) and in the README. None is presented as a recovered original run artifact.
