# Architecture / Reproduction Plan — SAM (arXiv:2010.01412)

**Paper:** [Foret et al. (2020), *Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2010_01412/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## SAM step

```text
g  = ∇L_B(w)
ε̂  = ρ · g / ‖g‖₂
g' = ∇L_B(w + ε̂)
w  ← SGD(w, g')
```

## Protocol

- CIFAR-10 WRN-28-10, 200 epochs, SAM (ρ=0.05) vs SGD with matched hyperparameters and seeds.

## Reproduction assumptions and limits

- Batch 256, LR 0.1 cosine, weight decay 5e-4, 200 epochs, rho 0.05.
- Emulate the paper's data-parallel m-sharpness by splitting each batch into chunks of m=128 with independent perturbations.
- Only the basic augmentation row (crop+flip) is implemented; Cutout and AutoAugment rows are not.
- No numeric targets are recorded in the SIR; read the target error rates from Table 1 when filling in results.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
