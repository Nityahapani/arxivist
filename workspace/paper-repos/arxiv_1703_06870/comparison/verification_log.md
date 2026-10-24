# Verification log — arXiv:1703.06870

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_1703_06870/sir.json` |
| RoIAlign exact on linear feature maps | PASS | `tests/test_mask_rcnn.py` |
| RoIAlign sensitive to sub-pixel shifts | PASS | `tests/test_mask_rcnn.py` |
| FPN level assignment | PASS | `tests/test_mask_rcnn.py` |
| Box coding round trip and NMS | PASS | `tests/test_mask_rcnn.py` |
| Mask loss gradients only on GT class | PASS | `tests/test_mask_rcnn.py` |
| End-to-end losses and inference | PASS | `tests/test_mask_rcnn.py` |
| R-50-FPN ≈ 44.4M parameters | PASS | `tests/test_mask_rcnn.py` |
| Pytest suite | PASS — 8 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny Mask R-CNN (width-8 ResNet-18 FPN) trained 4 steps on synthetic shape instances; finite outputs (params=256362, final_loss=2.713, mask_iou=0.159, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| COCO training and AP | NOT RUN | Requires 8 GPUs, COCO, pretrained backbone |
| Cityscapes / keypoints | NOT RUN | Not implemented |

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
PYTHONPATH=src python -m mask_rcnn_repro smoke --config configs/smoke.json --output-root results/smoke
```
