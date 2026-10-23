# Paper-to-reproduction comparison — arXiv:2111.06377

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| ViT-L/16 fine-tune top-1 | 85.9% | Pretraining implemented; fine-tuning not implemented; not run |
| ViT-H/14 448 px fine-tune | 87.8% | Not run |
| 75% optimal mask ratio | Fig. 5 | Configurable; ablation not run |
| ≥3× training speed-up | Encoder on 25% of tokens | Architecture implemented; timing not measured |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
