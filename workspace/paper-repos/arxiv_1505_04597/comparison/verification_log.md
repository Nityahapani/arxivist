# Verification log — arXiv:1505.04597

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_1505_04597/sir.json` |
| 572→388 output size | PASS | `tests/test_unet.py` |
| Weight map peaks in gaps between touching cells | PASS | `tests/test_unet.py` |
| Elastic deformation keeps label set | PASS | `tests/test_unet.py` |
| Overlap-tile reconstructs full image | PASS | `tests/test_unet.py` |
| Pytest suite | PASS — 8 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — depth-2 U-Net (8 base channels) trained 10 steps on synthetic touching cells with elastic deformation and the weight map, then overlap-tile evaluated; finite outputs (params=29330, output_size=52, eval_iou=0.1581, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ISBI 2012 EM challenge | NOT RUN | Requires dataset and server evaluation |
| ISBI 2015 cell tracking | NOT RUN | Requires dataset |

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
PYTHONPATH=src python -m unet_repro smoke --config configs/smoke.json --output-root results/smoke
```
