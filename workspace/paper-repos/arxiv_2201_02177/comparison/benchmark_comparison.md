# Paper-to-reproduction comparison — arXiv:2201.02177

This file distinguishes reported paper results from results actually run in this checkout. The full protocol is implemented in `configs/experiment_suite.json`, but the paper-scale grid has not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| Delayed generalization on modular division | At 50% training data, training accuracy becomes near-perfect in <1,000 updates; validation approaches the same level around 1,000,000 updates. | Protocol implemented (`figure1_modular_division`); not run at paper budget. |
| S5 data efficiency | Around 25–30% training data, a 1 percentage-point data reduction raises median updates to 99% validation accuracy by about 40–50%. | Seven-seed, 500k-update grid implemented (`figure1_s5_learning_time`); not run. |
| Optimizer and regularization ablations | Weight decay is especially effective; minibatch/noise/dropout and learning-rate variants are compared on S5. | All nine conditions implemented (`figure2_ablations`); not run at 100k updates. |
| Operation diversity | Best validation accuracy after 100k updates is reported for 12 modular and S5 operations, three seeds. | All 12 operation tables and multi-seed grid implemented (`figure2_variety`); not run. |
| Embedding structure | t-SNE shows structure in modular-addition and S5 output-layer symbol weights. | t-SNE pipeline implemented; representative checkpoints have not been trained. |
| Outlier robustness | k ∈ {0,10,100,1000,2000,3000}; small k has little effect, larger k narrows the generalizing range. | Corruption/data-efficiency pipeline implemented; operation choice is an explicit assumption (modular addition); not run. |
| Sharpness association | S5 validation accuracy and sharpness have Spearman rho = −0.79548, p < 0.000014. | Keskar-style low-dimensional sharpness and correlation code implemented; 20 paper-protocol checkpoints have not been trained or analyzed. |

The only executed training run is the 12-update CPU smoke test in `results/smoke/`; it verifies the training/output path, not any phenomenon or numeric result in the paper. Do not substitute smoke-test values for the paper column.
