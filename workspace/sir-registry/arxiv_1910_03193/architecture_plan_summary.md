# Architecture / Reproduction Plan — DeepONet (arXiv:1910.03193)

**Paper:** [Lu et al. (2019), *DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators*](https://arxiv.org/abs/1910.03193)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_1910_03193/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Core model

```text
u(x_1..x_m) ─► Branch MLP ─► b ∈ R^p ┐
                                      ├─► Σ_k b_k t_k + b_0 ─► G(u)(y)
y ──────────► Trunk MLP  ─► t ∈ R^p ┘
```

## Protocol

- GRF inputs (RBF kernel, l=0.2), m=100 sensors, reference solutions from trapezoid (antiderivative) or RK4.
- Train unstacked, stacked, and FNN with Adam (lr 1e-3) on MSE; report test MSE.

## Expected outputs

- Test MSE per model and operator; learning curves (train/test MSE vs step).

## Reproduction assumptions and limits

- Use p=40, branch hidden [40], trunk hidden [40, 40] with ReLU.
- 10,000 training (u, y) pairs with one output location per function; 1,000 test functions x 100 locations.
- 50,000 full-batch Adam steps at lr 1e-3.
- Pendulum operator integrated on T=1 using RK4 on a 1,001-point grid.
- The diffusion-reaction PDE example is not implemented in this scaffold.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
