# Paper-to-reproduction comparison — arXiv:1703.06870

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| X-101-FPN mask / box AP | 37.1 / 39.8 | Not run |
| R-101-FPN mask AP | 35.7 | Not run |
| RoIAlign vs RoIPool (Table 2c) | Large AP gain | Not run |
| Speed | 5 fps | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
