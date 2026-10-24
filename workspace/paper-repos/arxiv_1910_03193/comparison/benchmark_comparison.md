# Paper-to-reproduction comparison — arXiv:1910.03193

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| DeepONet vs FNN generalization | DeepONet has lower test error than FNN on the ODE operators | Both models implemented and trained on identical data by `configs/paper.json`; not run |
| Unstacked vs stacked | Unstacked + bias generalizes better | Both variants implemented; not run |
| Error vs data size | Fast convergence of test error with training-set size | Vary `n_train_functions`; not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
