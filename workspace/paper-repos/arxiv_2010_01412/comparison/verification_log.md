# Verification log — arXiv:2010.01412

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2010_01412/sir.json` |
| Ascent step perturbation has L2 norm ρ | PASS | `tests/test_sam.py` |
| Second step restores w and applies perturbed gradient | PASS | `tests/test_sam.py` |
| m-sharpness with m = batch equals standard SAM | PASS | `tests/test_sam.py` |
| WRN-28-10 ≈ 36.5M params | PASS | `tests/test_sam.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — WRN-10-1 trained 10 SAM steps (ρ=0.05) on synthetic 16×16 images; finite outputs (rho=0.05, test_acc=0.4062, test_error=0.5938, params=77460) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| CIFAR-10/100 WRN-28-10 SAM vs SGD | NOT RUN | Requires GPU; not executed |

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
PYTHONPATH=src python -m sam_repro smoke --config configs/smoke.json --output-root results/smoke
```
