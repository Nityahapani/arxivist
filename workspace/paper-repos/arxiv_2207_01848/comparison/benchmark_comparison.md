# Paper-to-reproduction comparison — arXiv:2207.01848

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Test predictions independent of other test points | Section 4 | Verified |
| Invariance to training order | Section 4 | Verified |
| ROC AUC on OpenML-CC18 vs GBDT / AutoML | Table 1 | Not run |
| 230× (CPU) / 5700× (GPU) speed-up | Section 5 | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
