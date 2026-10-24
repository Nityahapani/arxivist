# Paper-to-reproduction comparison — arXiv:2305.18290

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Loss at initialisation | log 2 | Verified |
| Optimal policy π* ∝ π_ref·exp(r/β) | Eq. 4 | Verified (bandit) |
| Reward/KL frontier vs PPO (Fig. 2) | DPO dominates | Not run |
| TL;DR / HH win rates (Fig. 3) | See paper | Not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
