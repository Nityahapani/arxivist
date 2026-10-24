# Verification log — arXiv:1910.03193

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_1910_03193/sir.json` |
| GRF samples are smooth / correlated as expected | PASS | `tests/test_data.py` |
| Reference solvers match closed forms (x², tanh) | PASS | `tests/test_data.py` |
| DeepONet output equals branch·trunk + b0; stacked has p branches | PASS | `tests/test_model.py` |
| Short training reduces test MSE | PASS | `tests/test_model.py` |
| Pytest suite | PASS — 9 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — unstacked DeepONet (p=10) trained 60 Adam steps on 64 antiderivative functions; finite outputs (params=561, final_train_mse=0.04576, final_test_mse=0.05101, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| Paper-scale operator experiments | NOT RUN | Not executed in the authoring environment; left to the assigned contributor |

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
PYTHONPATH=src python -m deeponet_repro smoke --config configs/smoke.json --output-root results/smoke
```
