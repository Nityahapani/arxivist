# Paper-to-reproduction comparison — arXiv:2006.07733

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Linear top-1 ResNet-50 | 74.3% | Not run |
| Linear top-1 ResNet-200 (2x) | 79.6% | Not run |
| Loss = 2 − 2cos | Eq. 2 | Verified |
| τ schedule endpoints | 0.996 → 1 | Verified |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
