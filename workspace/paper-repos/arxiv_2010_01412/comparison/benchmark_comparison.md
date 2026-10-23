# Paper-to-reproduction comparison — arXiv:2010.01412

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| WRN-28-10 CIFAR-10 test error, SAM vs SGD | SAM lower than SGD (see Table 1) | Both arms implemented; not run |
| CIFAR-100 improvement | SAM lower than SGD (see Table 1) | Set `data.name=cifar100`, `n_classes=100`; not run |
| m-sharpness correlation | Smaller m correlates better with generalization | m-sharpness step implemented; study not run |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
