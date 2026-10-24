# Verification log — arXiv:2201.03545

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2201_03545/sir.json` |
| ConvNeXt-T / S parameter counts equal 28.6M / 50.2M | PASS | `tests/test_convnext.py` |
| Stage resolutions /4, /8, /16, /32 | PASS | `tests/test_convnext.py` |
| Layer scale 1e-6 makes blocks near-identity at init | PASS | `tests/test_convnext.py` |
| Mixup/CutMix soft targets sum to 1 | PASS | `tests/test_convnext.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — debug ConvNeXt (depths 1-1-1-1, widths 16–128) trained 20 steps with Mixup/CutMix on synthetic 32×32 images; finite outputs (params=233044, top1=0.3047, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ImageNet-1K training (Table 1) | NOT RUN | Requires multi-GPU cluster and ImageNet |

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
PYTHONPATH=src python -m convnext_repro smoke --config configs/smoke.json --output-root results/smoke
```
