# Assumption and ambiguity report — arXiv:2307.08621

## Verified directly from the paper

- Eq. 1–8 and γ_i schedule.
- Value dimension 2d, swish gate, GroupNorm.

## Explicit assumptions in this reproduction

1. **Normalisations:** The paper's three retention normalisations (QK^T/sqrt(d), decay-matrix normalisation, score normalisation) are reduced to key scaling d^-1/2; GroupNorm's scale invariance makes the others numerically, not functionally, relevant. *Basis:* Keeps the three forms exactly equivalent in tests.
2. **Rotation:** xPos-style Θ is implemented as RoPE without the xPos length-extrapolation scaling. *Basis:* Paper describes Θ_n = e^{inθ}.
3. **Training hyperparameters:** AdamW (0.9, 0.98), wd 0.05, LR 6e-4, warmup 375, 25k steps for 1.3B. *Basis:* Read from Table 7 as best recalled; verify before paper-scale runs.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2307_08621/sir.json`) and in the README. None is presented as a recovered original run artifact.
