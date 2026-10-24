# Paper-to-reproduction comparison — arXiv:2304.07193

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| ViT-g/14 linear top-1 | 86.5% | Not run |
| ViT-S/14 parameters | 21M | Verified |
| Sinkhorn equipartition | Section 4 | Verified |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
