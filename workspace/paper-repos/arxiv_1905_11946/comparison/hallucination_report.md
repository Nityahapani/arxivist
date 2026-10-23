# Assumption and ambiguity report — arXiv:1905.11946

## Verified directly from the paper

- B0 architecture per Table 1 (stem 32, 7 MBConv stages, head 1280).
- α=1.2, β=1.1, γ=1.15 with α·β²·γ²≈2.
- Training: RMSProp decay 0.9 / momentum 0.9, BN momentum 0.99, wd 1e-5, LR 0.256 decaying 0.97 every 2.4 epochs, SiLU, AutoAugment, stochastic depth 0.8, dropout 0.2→0.5.

## Explicit assumptions in this reproduction

1. **Batch / epochs:** Batch 4096 and 350 epochs from the authors' public TPU recipe. *Basis:* Paper text does not state batch size or epochs.
2. **Label smoothing:** Label smoothing 0.1 (public recipe). *Basis:* Not mentioned in the paper.
3. **AutoAugment / EMA:** AutoAugment and weight EMA are not implemented in the scaffold. *Basis:* Scope limit; required for paper accuracy.
4. **SE reduction:** SE squeezes to 0.25 x block input channels. *Basis:* Matches the reference implementation; paper says 'squeeze-and-excitation optimization'.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_1905_11946/sir.json`) and in the README. None is presented as a recovered original run artifact.
