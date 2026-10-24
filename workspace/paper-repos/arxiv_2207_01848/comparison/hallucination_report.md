# Assumption and ambiguity report — arXiv:2207.01848

## Verified directly from the paper

- PFN objective.
- Architecture sizes.
- Dataset limits (1000 / 100 / 10).

## Explicit assumptions in this reproduction

1. **Prior:** A simplified SCM/BNN mixture (random MLP structure, random activations, node noise, quantile thresholds) without feature warping, categorical features or missing values. *Basis:* The paper's prior (Appendix C) has many more hyper-priors.
2. **Prior-fitting schedule:** Adam with warmup-cosine; steps, batch and LR are assumptions. *Basis:* Not confidently recovered.
3. **Ensembling:** Feature-permutation and class-relabelling ensemble; power transforms are omitted. *Basis:* Simplification.
4. **Evaluation:** Smoke evaluation on sklearn's breast cancer dataset instead of OpenML-CC18. *Basis:* Offline availability.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2207_01848/sir.json`) and in the README. None is presented as a recovered original run artifact.
