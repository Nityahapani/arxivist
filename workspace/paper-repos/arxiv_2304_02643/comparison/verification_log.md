# Verification log — arXiv:2304.02643

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2304_02643/sir.json` |
| Window partition roundtrip | PASS | `tests/test_segment_anything.py` |
| Global blocks equally spaced | PASS | `tests/test_segment_anything.py` |
| Prompt padding and shapes | PASS | `tests/test_segment_anything.py` |
| Multimask and single-mask outputs | PASS | `tests/test_segment_anything.py` |
| Focal and dice values | PASS | `tests/test_segment_anything.py` |
| Minimum-over-masks loss | PASS | `tests/test_segment_anything.py` |
| Corrective points in error region | PASS | `tests/test_segment_anything.py` |
| Pytest suite | PASS — 9 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny SAM (64 px, 4-block windowed ViT) trained 4 steps on synthetic shapes with 3 interactive rounds per mask; finite outputs (params=342476, final_loss=3.141, final_mask_iou=0.02787, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| SA-1B training and data engine | NOT RUN | Requires dataset and cluster |
| Zero-shot transfer benchmarks | NOT RUN | Requires datasets and trained weights |

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
PYTHONPATH=src python -m segment_anything_repro smoke --config configs/smoke.json --output-root results/smoke
```
