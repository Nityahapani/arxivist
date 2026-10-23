# Verification log — arXiv:1901.08573

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_1901_08573/sir.json` |
| WRN-34-10 parameter count ≈ 46.2M | PASS | `tests/test_trades.py` |
| Adversarial examples stay inside ε-ball and [0,1] | PASS | `tests/test_trades.py` |
| TRADES loss = CE + β·KL | PASS | `tests/test_trades.py` |
| PGD does not decrease CE | PASS | `tests/test_trades.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — WRN-10-1 trained 6 TRADES steps (2-step inner max) on synthetic 16×16 images, then 3-step PGD evaluation; finite outputs (natural_acc=0.2969, robust_acc_pgd=0.2969, beta=6, params=77460) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| CIFAR-10 WRN-34-10 training (Table 5) | NOT RUN | Requires GPU-days; not executed |

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
PYTHONPATH=src python -m trades_repro smoke --config configs/smoke.json --output-root results/smoke
```
