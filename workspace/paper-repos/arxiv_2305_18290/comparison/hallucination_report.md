# Assumption and ambiguity report — arXiv:2305.18290

## Verified directly from the paper

- Eq. 1, 4, 7 and the gradient expression.
- β=0.1 default, RMSprop lr 1e-6, batch 64, 150 warmup steps.

## Explicit assumptions in this reproduction

1. **Reported numbers:** No numeric targets are hard-coded; read targets from Figure 2 (reward/KL frontier), Figure 3 (TL;DR and HH win rates) and Table 2 of the paper. *Basis:* Results are mostly reported as plots.
2. **Synthetic task:** A synthetic controlled-generation task with a known reward stands in for IMDb + sentiment classifier. *Basis:* Allows exact ground-truth reward and CPU smoke tests.
3. **SFT data:** Smoke SFT uses uniform random tokens. *Basis:* Paper uses in-domain SFT or Preferred-FT.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2305_18290/sir.json`) and in the README. None is presented as a recovered original run artifact.
