# Verification log — arXiv:2103.14030

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2103_14030/sir.json` |
| Window partition/reverse round trip | PASS | `tests/test_swin.py` |
| Relative position index range | PASS | `tests/test_swin.py` |
| Shift mask blocks cross-region attention (token perturbation test) | PASS | `tests/test_swin.py` |
| Swin-T parameter count = 28.3M | PASS | `tests/test_swin.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — debug Swin (C=24, window 4) trained 10 steps on synthetic 64×64 images; finite outputs (params=1285552, top1=0.25, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ImageNet-1K training | NOT RUN | Requires multi-GPU cluster and ImageNet |
| COCO / ADE20K | NOT RUN | Not implemented |

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
PYTHONPATH=src python -m swin_repro smoke --config configs/smoke.json --output-root results/smoke
```
