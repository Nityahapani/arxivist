# Paper-to-reproduction comparison — arXiv:2205.14135

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Exact attention output and gradients | Section 3 | Verified |
| Fewer HBM accesses | Theorem 2 | Verified analytically |
| BERT-large 15% / GPT-2 3× / LRA 2.4× | Section 4 | Not run |
| Path-X 61.4% / Path-256 63.1% | Table 6 | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
