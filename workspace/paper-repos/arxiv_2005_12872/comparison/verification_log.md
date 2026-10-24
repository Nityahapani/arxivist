# Verification log — arXiv:2005.12872

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2005_12872/sir.json` |
| GIoU properties | PASS | `tests/test_detr.py` |
| Matcher optimal vs brute force | PASS | `tests/test_detr.py` |
| Loss invariant to target order | PASS | `tests/test_detr.py` |
| No-object weight 0.1 | PASS | `tests/test_detr.py` |
| Output shapes, boxes in [0,1] | PASS | `tests/test_detr.py` |
| R50 ≈ 41M parameters | PASS | `tests/test_detr.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny DETR (width-8 ResNet-18, d=32) trained 6 steps on synthetic coloured rectangles; finite outputs (params=215760, final_loss=16.32, recall_iou50=0, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| COCO 300/500-epoch training | NOT RUN | Requires 16 GPUs and COCO |
| Panoptic segmentation | NOT RUN | Not implemented |

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
PYTHONPATH=src python -m detr_repro smoke --config configs/smoke.json --output-root results/smoke
```
