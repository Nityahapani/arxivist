# Assumption and ambiguity report — arXiv:1505.04597

## Verified directly from the paper

- Fig. 1 layer configuration and tile sizes.
- Eq. 2 with w0=10, σ≈5.
- Elastic deformation: 3×3 grid, 10 px std.
- SGD momentum 0.99, batch of a single image.

## Explicit assumptions in this reproduction

1. **Learning rate / iterations:** LR 0.01 and 10k iterations. *Basis:* Not stated in the paper.
2. **Weight map on background:** The border term is applied to background pixels between cells. *Basis:* Eq. 2 is motivated by separating borders; reference Caffe implementation is not public as code here.
3. **Elastic interpolation:** Coarse 3x3 displacement field upsampled with cubic spline zoom (scipy). *Basis:* Paper says bicubic interpolation.
4. **Smoke data:** Synthetic touching disks replace EM/cell data for CI. *Basis:* Datasets require registration/manual download.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_1505_04597/sir.json`) and in the README. None is presented as a recovered original run artifact.
