# Assumption and ambiguity report — arXiv:1910.03193

## Verified directly from the paper

- Branch net takes u at m fixed sensors; trunk net takes y; output is their inner product (plus bias b_0).
- Stacked (p branch nets) and unstacked (one branch net) variants are defined.
- Input functions are sampled from Gaussian random fields.

## Explicit assumptions in this reproduction

1. **Network widths:** Use p=40, branch hidden [40], trunk hidden [40, 40] with ReLU. *Basis:* Chosen as representative small widths; exact widths per experiment must be checked against the paper's tables.
2. **Dataset sizes:** 10,000 training (u, y) pairs with one output location per function; 1,000 test functions x 100 locations. *Basis:* Exact sample counts differ across experiments in the paper.
3. **Training budget:** 50,000 full-batch Adam steps at lr 1e-3. *Basis:* Iteration counts are not uniformly stated.
4. **Pendulum horizon:** Pendulum operator integrated on T=1 using RK4 on a 1,001-point grid. *Basis:* Implementation choice for the reference solver.
5. **Reaction-diffusion:** The diffusion-reaction PDE example is not implemented in this scaffold. *Basis:* Scope limit; left for the contributor.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_1910_03193/sir.json`) and in the README. None is presented as a recovered original run artifact.
