# Assumption and ambiguity report — arXiv:2201.02177

## Verified directly from the paper

- The studied examples are finite binary-operation equations with each symbol tokenized separately.
- The model is a 2-layer decoder-only Transformer with width 128 and 4 heads, about 4×10⁵ non-embedding parameters.
- Default AdamW settings include lr 1e-3, weight decay 1, β=(0.9,0.98), 10-update linear warmup, and batch size 512 or half the training set, whichever is smaller.
- The paper covers nine modular operations and three S5 operations, with the precise formulas listed in Appendix A.1.1.
- Sections 3.1, 3.1.1 and 3.2/3.3 use different budgets/seeds; Appendix A.1.2 records 1e6, 5e5 and 1e5 update budgets and the 3/7-seed distinction.

## Explicit assumptions in this reproduction

1. **S5 composition convention:** `(x*y)[i] = y[x[i]]`, matching the authors' public data code; the paper itself does not define array composition.
2. **EOS loss:** the code scores the answer token only, following the paper prose. The public author implementation's right-side target slice includes a trailing EOS token as well.
3. **Fractions:** 5-percentage-point grids with denser sampling near 25–32% for the S5 time curve; the paper does not enumerate every fraction.
4. **Figure 6 task:** modular addition is used for the outlier experiment because the article text/caption does not name the operation.
5. **Figure 3 checkpoints and t-SNE:** seed 0, 50% training fraction, PCA initialization, perplexity 30, random state 0; checkpoint details are not fully given in the paper.
6. **Figure 7 checkpoint:** S5 product at 70% data, 100k updates, seeds 0–19. The public sharpness script lists T-70 and seeds 0–19; the paper does not state the exact fixed update count.

These assumptions are also documented in the SIR and the main README. None is presented as a recovered original run artifact. The exact version is `workspace/sir-registry/arxiv_2201_02177/sir.json`.
