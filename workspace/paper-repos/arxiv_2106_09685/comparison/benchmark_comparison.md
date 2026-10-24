# Paper-to-reproduction comparison — arXiv:2106.09685

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| RoBERTa-base GLUE average | 87.2 (LoRA, 0.3M trainable params) | Protocol implemented (`configs/paper.json`); not run |
| MNLI / SST-2 (RoBERTa-base) | 87.5 / 95.1 | Not run |
| Parameter efficiency | Up to 10,000x fewer trainable params (GPT-3 175B) | Parameter-count formula verified in tests; GPT-3 not in scope |
| No inference latency | BA merged into W0 | Merge/unmerge equivalence verified in tests (max abs diff < 1e-5) |
| GPT-2 M E2E BLEU | 70.4 | Not implemented |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
