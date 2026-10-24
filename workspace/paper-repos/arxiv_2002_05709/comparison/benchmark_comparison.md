# Paper-to-reproduction comparison — arXiv:2002.05709

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| ImageNet linear eval, ResNet-50 | 69.3% top-1 | Not run (requires ImageNet + cluster) |
| ImageNet linear eval, ResNet-50 (4x) | 76.5% top-1 | Not run |
| CIFAR-10 linear eval (Appendix B.9) | See paper Appendix B.9 | Protocol implemented (`configs/paper.json`); not run |
| Projection head helps | Nonlinear head > linear > none | Head implemented; ablation not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
