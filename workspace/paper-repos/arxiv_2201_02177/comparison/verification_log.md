# Verification log — arXiv:2201.02177

**Date:** 2026-10-23  
**Status:** Implementation and smoke verification complete; paper-scale experiment matrix not run.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2201_02177/sir.json` |
| All 12 Appendix A.1.1 operations and table sizes | PASS | `tests/test_operations.py` |
| Random split, tokenization, modular division domain, outlier-only-train behavior | PASS | `tests/test_data.py` |
| Paper-sized Transformer output, parameter count, causal masking | PASS | `tests/test_model.py` |
| Warmup and short optimizer/training/checkpoint path | PASS | `tests/test_training.py` |
| Pytest suite | PASS — 15 tests | `pytest -q tests/` |
| CPU end-to-end smoke run | PASS — 12 updates on modular addition, 50% split; finite losses/accuracies and CSV/JSON outputs | `results/smoke/smoke/mod_add_frac-0p5_seed-0_default_k-0/` |
| Paper-sized backward-pass smoke | PASS — 1 update, width 128 / 2 layers / 4 heads, 394,240 non-embedding parameters | Truncated `figure1_modular_division` run; not a paper result |
| Figure 3 t-SNE and division-table export helpers | PASS | Two tiny, one-step temporary checkpoints; temporary files removed |
| Figure 1–7 full training grids | NOT RUN | No GPU in the authoring environment; full manifest is approximately 1,456 runs / 198.7M optimizer updates |

The smoke configuration uses a 1-layer, width-32 model for 12 updates. Its near-chance accuracy is expected and is not a paper result. The paper-sized model is separately tested for tensor shapes and parameter count. No headline metric or paper figure is claimed as reproduced until the complete relevant suite is run.

## Environment used for checks

- Python 3.13.16
- PyTorch 2.14.1+cpu
- CPU-only runtime (2 visible CPU cores; no CUDA device)
- NumPy 2.3.5, SciPy 1.17.1, scikit-learn 1.6.1, Matplotlib 3.10.9

## Repeatable commands

From the repository root:

```bash
pytest -q
PYTHONPATH=src python -m grokking_repro run \
  --config configs/smoke.json --suite smoke --output-root results/smoke
```
