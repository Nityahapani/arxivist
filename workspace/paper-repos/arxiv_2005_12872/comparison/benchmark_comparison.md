# Paper-to-reproduction comparison — arXiv:2005.12872

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| DETR R50 AP | 42.0 | Not run |
| DETR-DC5 R50 AP | 43.3 | Not run |
| DETR R101 AP | 43.5 | Not run |
| R50 parameters | 41M | Verified |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
