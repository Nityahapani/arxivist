# Paper-to-reproduction comparison — arXiv:2108.12409

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Extrapolation (WikiText-103) | ALiBi perplexity stays flat/improves for L_valid > L_train; sinusoidal degrades | Protocol implemented; not run |
| 1.3B efficiency | 11% faster, 11% less memory at matched perplexity | Out of scope |
| Slopes | Geometric 2^{-8h/n} | Verified in tests |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
