# Paper-to-reproduction comparison — arXiv:2307.08621

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Parallel = recurrent = chunkwise | Section 2.1 | Verified |
| Decoding speed-up at 7B, 8k | 8.4× | Not run (toy benchmark only) |
| Decoding memory saving | 70% | Not run |
| Perplexity scaling vs Transformer | Fig. 5 | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
