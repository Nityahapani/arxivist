# Paper-to-reproduction comparison — arXiv:2405.21060

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Recurrent = quadratic = chunked | Sections 4–6 | Verified |
| SSD speed-up vs Mamba scan | 2–8× | Not run (needs GPU kernels) |
| MQAR accuracy vs Mamba / attention | Fig. 8 | Not run |
| LM scaling | Fig. 9 | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
