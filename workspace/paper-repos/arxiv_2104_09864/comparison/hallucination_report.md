# Assumption and ambiguity report — arXiv:2104.09864

## Verified directly from the paper

- Eq. 15 rotation matrix and θ_i schedule.
- Eq. 34 efficient implementation.
- Eq. 19 linear attention form.

## Explicit assumptions in this reproduction

1. **Reported numbers:** No numeric targets are hard-coded; read targets from Table 2 (GLUE), Table 5 (CAIL2019-SCM) and Figures 3–4 (pretraining curves). *Basis:* Avoid citing uncertain numbers.
2. **Pair layout:** Interleaved pairs (x_{2i-1}, x_{2i}) as in Eq. 15. *Basis:* Many implementations use a half-split layout; both are equivalent up to a permutation.
3. **Linear-attention feature map:** phi = elu + 1 for both queries and keys. *Basis:* Paper leaves the non-negative feature map generic.
4. **Training hyperparameters:** AdamW, LR 1e-4, 10k warmup in configs/paper.json. *Basis:* Not confidently recovered from the paper.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2104_09864/sir.json`) and in the README. None is presented as a recovered original run artifact.
