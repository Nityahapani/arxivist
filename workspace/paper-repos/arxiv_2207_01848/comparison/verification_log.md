# Verification log — arXiv:2207.01848

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2207_01848/sir.json` |
| Attention mask | PASS | `tests/test_tabpfn.py` |
| Test-point independence | PASS | `tests/test_tabpfn.py` |
| Training-order invariance | PASS | `tests/test_tabpfn.py` |
| Prior sample shapes / padding | PASS | `tests/test_tabpfn.py` |
| Ensemble probabilities | PASS | `tests/test_tabpfn.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer PFN prior-fitted for 10 steps on synthetic SCM/BNN datasets, then evaluated zero-shot on breast cancer; finite outputs (params=20644, final_prior_nll=1.358, tabpfn_auc=0.2295, tabpfn_acc=0.59) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| Full prior fitting | NOT RUN | Requires multi-GPU days |
| OpenML-CC18 benchmark | NOT RUN | Requires OpenML data and baselines |

The smoke configuration is deliberately tiny (see `configs/smoke.json`); its metrics are expected to be far from the paper and are **not** paper results. No headline metric or paper figure is claimed as reproduced until the corresponding paper-scale run is completed and recorded in `benchmark_comparison.md`.

## Environment used for checks

- Python 3.13.16
- PyTorch 2.14.1+cpu
- NumPy 2.5.4
- CPU-only runtime (2 visible CPU cores; no CUDA device)

## Repeatable commands

From this directory:

```bash
pytest -q
PYTHONPATH=src python -m tabpfn_repro smoke --config configs/smoke.json --output-root results/smoke
```
