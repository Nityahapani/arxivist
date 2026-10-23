# Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets

**Paper:** Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra (2022), [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).
**Code:** independent PyTorch reproduction, informed by the paper and the authors' public [`openai/grok`](https://github.com/openai/grok) implementation. This is not the authors' repository and does not claim access to their original run logs.

## Reproduction status

This repository implements the paper's finite-operation datasets, 2-layer Transformer, optimizer/regularization variants, multi-seed experiment grids, plotting, t-SNE, and sharpness analysis. It includes unit tests and a short end-to-end smoke configuration.

**The full paper-scale grid has not been run here.** This workspace has no GPU; the full manifest contains 1,456 runs and roughly 199 million optimizer updates (including the 1M- and 500k-step studies). The smoke run and tests verify code paths, not the paper's empirical claims. Do not interpret smoke metrics as a reproduction result. A complete run needs a CUDA-capable machine or an appropriately provisioned compute cluster.

## What the paper studies

The task is to learn a small binary-operation table. Each equation is presented as separate symbol tokens, for example `x op y = answer`. A small decoder-only Transformer can reach almost perfect training accuracy first, remain near chance on held-out equations for much longer, and then generalize suddenly. The paper also varies the amount of training data, compares optimizer and regularization choices, visualizes learned output-layer symbol weights, adds mislabeled outliers, and measures local sharpness.

## Paper coverage

| Paper section / figure | Suite or command | Implemented coverage |
|---|---|---|
| Figure 1, left; Section 3.1 | `figure1_modular_division` | Division mod 97, 50% training fraction, Adam without weight decay, 1M updates, three seeds; train/validation accuracy and loss curves |
| Figure 1, center; Section 3.1.1 | `figure1_s5_learning_time` | S5 product, 500k-update budget and seven seeds; first logged update reaching 99% validation accuracy, median by data fraction |
| Figure 2, left; Section 3.3 | `figure2_ablations` | All nine reported optimizer / regularization conditions on S5 product, three seeds, 100k updates |
| Figure 2, right; Section 3.2 | `figure2_variety` | All 12 Appendix A.1.1 operations, three seeds, 100k updates; best validation accuracy by data fraction |
| Figure 3; Section 3.4 | `figure3_embeddings` | Representative modular-addition and S5 models; t-SNE of output/unembedding rows |
| Figure 4; Appendix A.2 | `figure1_modular_division` + `plot` | Train and validation cross-entropy curves for modular division |
| Figure 5; Appendix A.2 | `table` | Generated truth-table CSV export; the paper does not identify which operation or exact symbol assignment is shown, so this is Figure-5-style output, not a claim of exact pixel/table recovery |
| Figure 6; Appendix A.4 | `appendix_a4_outliers` | k ∈ {0, 10, 100, 1000, 2000, 3000}; train labels corrupted, validation labels left clean |
| Figure 7; Appendix A.5 | `appendix_a5_sharpness` + `sharpness` | 20 S5-product initializations at 70% data; Keskar-style low-dimensional sharpness and Spearman correlation |

The complete operation inventory is in `configs/experiment_suite.json` and `src/grokking_repro/operations.py`:

1. x+y (mod 97)
2. x−y (mod 97)
3. x/y (mod 97), y≠0
4. x/y for odd y, otherwise x−y (mod 97)
5. x²+y² (mod 97)
6. x²+xy+y² (mod 97)
7. x²+xy+y²+x (mod 97)
8. x³+xy (mod 97)
9. x³+xy²+y (mod 97)
10. S5 product
11. S5 conjugation x·y·x⁻¹
12. S5 x·y·x

## Model and training defaults

- **Model:** decoder-only Transformer; 2 layers, width 128, 4 heads, head width 32; sinusoidal positional encoding; causal self-attention; 4×-width ReLU feed-forward block; post-residual LayerNorm; about 394k non-embedding parameters in this implementation.
- **Inputs:** EOS prefix plus the four equation tokens before the answer; the output at the `=` position predicts one atomic answer token. The loss and exact accuracy are answer-token-only, matching the paper prose.
- **Dropout ablation:** residual dropout is applied to the feed-forward branch and not the attention-output branch, matching the authors' public implementation (whose attention-dropout line is disabled).
- **Default optimizer:** AdamW, learning rate 1e-3, weight decay 1, β=(0.9, 0.98), 10-update linear warmup, then constant LR; minibatch size `min(512, ceil(n_train/2))`.
- **No annealing:** the paper notes annealing sometimes helped but uses a constant post-warmup rate for its final configuration.
- **Metrics:** answer-token cross-entropy and exact accuracy on train and validation sets; the first *logged* update reaching 99% validation accuracy; best validation accuracy within the run budget.

## Important assumptions and limits

The paper does not specify every sampled data fraction, name the binary operation used for Figure 6, state the exact Figure 3 checkpoints or t-SNE hyperparameters, or give an exact step count for its sharpness checkpoint. The manifest records explicit choices instead of presenting them as recovered facts:

- Data-fraction curves use 5-percentage-point grids from 5% to 95%; the S5 time curve is denser around 25–32%.
- Figure 6 uses modular addition because the paper text/caption does not name its operation; the same code supports changing it to S5 product or modular division.
- Representative t-SNE checkpoints use seed 0 and a 50% train split. t-SNE uses PCA initialization, perplexity 30, and random state 0.
- Figure 7 uses S5 product, 70% training data, 100k updates, and seeds 0–19. The public author sharpness script also lists S5 checkpoints with T-70 and seeds 0–19; the fixed step is an explicit choice because the paper only says to stop when about half the models generalize.
- S5 permutations use the public author-code array convention `(x*y)[i] = y[x[i]]`; the article itself does not specify that convention.
- The paper says answer-only loss. The authors' public EOS-wrapped code appears to include the trailing EOS token in its right-side shifted-target slice; this implementation follows the paper prose and does not train on that extra EOS target.

See [`../sir-registry/arxiv_2201_02177/sir.json`](../../sir-registry/arxiv_2201_02177/sir.json) and the SIR [architecture plan](../../sir-registry/arxiv_2201_02177/architecture_plan_summary.md) for structured assumptions and confidence annotations.

## Install

Python 3.10+ is required. For CPU-only PyTorch, install the CPU wheel first; then install the project and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q
```

For CUDA, install the PyTorch build matching the machine's CUDA driver using the official PyTorch installation selector, then install this project with `python -m pip install -e '.[test]'`.

## Quick smoke test

This small-width, 12-step configuration checks the pipeline only; it is **not** a paper result.

```bash
PYTHONPATH=src python -m grokking_repro run \
  --config configs/smoke.json \
  --suite smoke \
  --output-root results/smoke
```

Or run the test suite:

```bash
pytest -q
```

## Run the paper-scale experiments

Run an individual suite, for example the main Figure 1 division curve:

```bash
PYTHONPATH=src python -m grokking_repro run \
  --config configs/experiment_suite.json \
  --suite figure1_modular_division \
  --output-root results/runs
```

Run all suites (compute-intensive; not suitable for a CPU smoke check):

```bash
PYTHONPATH=src python -m grokking_repro run-all \
  --config configs/experiment_suite.json \
  --output-root results/runs
```

`--steps N` and `--limit N` are provided for debugging. Any run made with these overrides is budget-truncated and must not be reported as the paper protocol. Each completed run writes `run_config.json`, `split_metadata.json`, `tokenizer.json`, `metrics.csv`, and `summary.json`; embedding and sharpness suites also write local `final.pt` checkpoints.

### Figures and analysis

```bash
# Figures 1, 2, 4, and 6 from completed CSV runs
PYTHONPATH=src python -m grokking_repro plot \
  --results-root results/runs --output-dir results/figures

# Also make Figure 3 (requires both Figure 3 final.pt checkpoints)
PYTHONPATH=src python -m grokking_repro plot \
  --results-root results/runs --output-dir results/figures --include-figure3

# Export a full 97 x 97 modular-addition table (Figure 5-style artifact)
PYTHONPATH=src python -m grokking_repro table \
  --operation mod_add --output results/figures/mod_add_truth_table.csv

# After training Appendix A.5 checkpoints, calculate sharpness and correlation
PYTHONPATH=src python -m grokking_repro sharpness \
  --checkpoints results/runs/appendix_a5_sharpness \
  --output-csv results/runs/appendix_a5_sharpness/sharpness.csv
```

## Layout

```text
configs/experiment_suite.json    # paper-scale multi-seed protocol manifest
configs/smoke.json              # deliberately small test configuration
src/grokking_repro/operations.py # 12 finite binary-operation tables
src/grokking_repro/data.py       # equation splits, atomic tokenization, outliers
src/grokking_repro/model.py      # 2-layer causal Transformer
src/grokking_repro/optim.py      # Adam/AdamW, decay-to-init and noise variants
src/grokking_repro/train.py      # reproducible single-run trainer and metrics
src/grokking_repro/runner.py     # experiment-suite expansion and execution
src/grokking_repro/sharpness.py  # Appendix A.5 sharpness approximation
src/grokking_repro/plotting.py   # paper figures, t-SNE, operation tables
```

## Citation

```bibtex
@article{power2022grokking,
  title={Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```
