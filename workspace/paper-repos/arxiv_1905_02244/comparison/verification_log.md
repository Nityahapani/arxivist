# Verification log — arXiv:1905.02244

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_1905_02244/sir.json` |
| h-swish matches definition | PASS | `tests/test_mobilenetv3.py` |
| Large / Small params = 5.4M / 2.5M | PASS | `tests/test_mobilenetv3.py` |
| Output stride 32 (960×7×7 at 224 px) | PASS | `tests/test_mobilenetv3.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — MobileNetV3-Small ×0.35 trained 10 RMSProp steps with EMA on synthetic 32×32 images; finite outputs (params=338460, top1=0.3125, ema_top1=0.25, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ImageNet training | NOT RUN | Requires multi-GPU and ImageNet |
| Mobile latency | NOT RUN | Requires Pixel device benchmarking |

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
PYTHONPATH=src python -m mobilenetv3_repro smoke --config configs/smoke.json --output-root results/smoke
```
