# Paper-to-reproduction comparison — arXiv:2305.14314

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| NF4 levels | Published table | Verified |
| DQ overhead | 0.127 bits/param | Verified |
| NF4 < FP4, Int4 error on normal weights | Section 3 / Fig. 3 | Verified (MSE) |
| Guanaco 65B vs ChatGPT | 99.3% | Not run |
| 65B finetuning on one 48GB GPU in 24h | Section 5 | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
