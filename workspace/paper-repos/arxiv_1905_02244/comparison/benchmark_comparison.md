# Paper-to-reproduction comparison — arXiv:1905.02244

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Large 1.0 top-1 / params | 75.2% / 5.4M | Params verified; accuracy not run |
| Small 1.0 top-1 / params | 67.4% / 2.5M | Params verified; accuracy not run |
| Latency vs MobileNetV2 | −20% (Large) | Requires Pixel phone benchmarking; not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
