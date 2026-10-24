# Verification log — arXiv:2304.07193

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2304_07193/sir.json` |
| Sinkhorn marginals | PASS | `tests/test_dinov2.py` |
| KoLeo prefers spread features | PASS | `tests/test_dinov2.py` |
| DINO loss skips same view | PASS | `tests/test_dinov2.py` |
| iBOT gradients only at masked patches | PASS | `tests/test_dinov2.py` |
| ViT-S/14 params and pos interpolation | PASS | `tests/test_dinov2.py` |
| Mask token replaces patch content | PASS | `tests/test_dinov2.py` |
| Teacher EMA and schedules | PASS | `tests/test_dinov2.py` |
| Mask ratios | PASS | `tests/test_dinov2.py` |
| Pytest suite | PASS — 9 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer ViT student/teacher trained 4 steps on synthetic images with 2 global + 2 local crops, then teacher k-NN evaluated; finite outputs (backbone_params=32384, final_loss=9.836, knn_top1=0.25, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| LVD-142M curation and ViT-g pretraining | NOT RUN | Requires private data and large cluster |
| Downstream evaluations | NOT RUN | Requires datasets and GPUs |

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
PYTHONPATH=src python -m dinov2_repro smoke --config configs/smoke.json --output-root results/smoke
```
