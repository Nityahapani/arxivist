# Paper-to-reproduction comparison — arXiv:2104.09864

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Relative-position property | Eq. 16 | Verified |
| Long-term decay | Fig. 2 | Verified qualitatively |
| Faster MLM convergence than BERT | Fig. 3 | Not run at scale |
| GLUE / CAIL2019-SCM | Tables 2, 5 | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
