# Assumption and ambiguity report — arXiv:2103.14030

## Verified directly from the paper

- Window size M=7, patch size 4, C=96 and depths (2,2,6,2) for Swin-T.
- Shifted windows displaced by ⌊M/2⌋ using cyclic shift + masking.
- Relative position bias with a (2M−1)×(2M−1) table.

## Explicit assumptions in this reproduction

1. **Augmentations:** RandAugment and Random Erasing are not implemented; Mixup/CutMix/label smoothing are. *Basis:* Scope limit; must be added before reporting.
2. **Small feature maps:** If a stage's resolution is <= window size, the window equals the full map and no shift is applied. *Basis:* Follows the reference implementation.
3. **Gradient clipping:** Clip-norm 5.0 from the reference recipe is documented but not applied in the scaffold loop. *Basis:* To be added with the full recipe.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2103_14030/sir.json`) and in the README. None is presented as a recovered original run artifact.
