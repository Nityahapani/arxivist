# Paper-to-reproduction comparison — arXiv:1901.08573

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Natural accuracy, WRN-34-10, 1/λ=6 | 84.92% | Protocol implemented; not run |
| Robust accuracy FGSM^k, WRN-34-10, 1/λ=6 | 56.61% | Protocol implemented; not run |
| Trade-off controlled by λ | Larger 1/λ → more robust, less accurate | β configurable; sweep not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
