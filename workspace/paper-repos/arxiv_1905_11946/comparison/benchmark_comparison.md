# Paper-to-reproduction comparison — arXiv:1905.11946

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| B0 top-1 / params | 77.1% / 5.3M | Params verified (5.3M); accuracy not run |
| B1 / B3 params | 7.8M / 12M | Verified |
| B7 top-1 / params | 84.3% / 66M | Not run |
| Compound vs single-dimension scaling (Fig. 8) | Compound best | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
