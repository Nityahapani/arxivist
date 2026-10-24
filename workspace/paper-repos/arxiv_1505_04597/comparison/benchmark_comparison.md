# Paper-to-reproduction comparison — arXiv:1505.04597

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| EM warping error | 0.000353 | Not run |
| PhC-U373 IoU | 0.9203 | Not run |
| DIC-HeLa IoU | 0.7756 | Not run |
| 572→388 tile geometry | Fig. 1 | Verified |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
