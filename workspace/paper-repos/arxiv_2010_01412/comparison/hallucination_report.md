# Assumption and ambiguity report — arXiv:2010.01412

## Verified directly from the paper

- ε̂(w) = ρ·g/‖g‖₂ for p=2.
- Gradient evaluated at w+ε̂ is used for the update; second-order terms dropped.
- One extra forward/backward pass per step.

## Explicit assumptions in this reproduction

1. **Hyperparameters:** Batch 256, LR 0.1 cosine, weight decay 5e-4, 200 epochs, rho 0.05. *Basis:* Representative CIFAR settings; contributors should verify against the paper's appendix table.
2. **m-sharpness:** Emulate the paper's data-parallel m-sharpness by splitting each batch into chunks of m=128 with independent perturbations. *Basis:* The paper notes that per-accelerator perturbation effectively computes m-sharpness.
3. **Augmentation:** Only the basic augmentation row (crop+flip) is implemented; Cutout and AutoAugment rows are not. *Basis:* Scope limit.
4. **Reported numbers:** No numeric targets are recorded in the SIR; read the target error rates from Table 1 when filling in results. *Basis:* Avoids transcribing values that could not be verified in the authoring environment.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2010_01412/sir.json`) and in the README. None is presented as a recovered original run artifact.
