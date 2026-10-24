# Verification log — arXiv:1905.11946

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_1905_11946/sir.json` |
| B0 / B1 / B3 parameter counts match Table 2 | PASS | `tests/test_efficientnet.py` |
| α·β²·γ² ≈ 2 | PASS | `tests/test_efficientnet.py` |
| Channel / repeat rounding | PASS | `tests/test_efficientnet.py` |
| MBConv residual and strided shapes | PASS | `tests/test_efficientnet.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — width-0.25 / depth-0.5 EfficientNet trained 10 RMSProp steps on synthetic 32×32 images; finite outputs (train_resolution=224, params=200106, top1=0.3125, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ImageNet training (Table 2) | NOT RUN | Requires multi-GPU cluster and ImageNet |

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
PYTHONPATH=src python -m efficientnet_repro smoke --config configs/smoke.json --output-root results/smoke
```
