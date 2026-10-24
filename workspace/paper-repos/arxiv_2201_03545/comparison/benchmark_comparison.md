# Paper-to-reproduction comparison — arXiv:2201.03545

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| ConvNeXt-T top-1 | 82.1% | Not run |
| ConvNeXt-T params | 28.6M | Verified: 28.6M (tests) |
| ConvNeXt-S params | 50.2M | Verified: 50.2M (tests) |
| ConvNeXt-XL 22K→1K top-1 (384) | 87.8% | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
