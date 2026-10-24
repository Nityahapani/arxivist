# Verification log — arXiv:2006.07733

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2006_07733/sir.json` |
| Loss equals 2 − 2cos | PASS | `tests/test_byol.py` |
| τ schedule endpoints | PASS | `tests/test_byol.py` |
| Target receives no gradients | PASS | `tests/test_byol.py` |
| EMA update rule | PASS | `tests/test_byol.py` |
| Asymmetric augmentation probabilities | PASS | `tests/test_byol.py` |
| Pytest suite | PASS — 8 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — BYOL with a width-8 ResNet-18 trained 8 LARS steps on synthetic 32×32 images, followed by a linear probe; finite outputs (final_byol_loss=3.151, linear_eval_top1=0.4219, encoder_params=175752, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ImageNet 1000-epoch pretraining and linear evaluation | NOT RUN | Requires TPU/GPU cluster and ImageNet |

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
PYTHONPATH=src python -m byol_repro smoke --config configs/smoke.json --output-root results/smoke
```
