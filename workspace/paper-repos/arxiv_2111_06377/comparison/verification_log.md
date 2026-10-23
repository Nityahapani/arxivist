# Verification log — arXiv:2111.06377

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2111_06377/sir.json` |
| Patchify/unpatchify round trip | PASS | `tests/test_mae.py` |
| Masking keeps exactly 25% and restore order is correct | PASS | `tests/test_mae.py` |
| Loss ignores visible patches | PASS | `tests/test_mae.py` |
| ViT-B/16 encoder ≈ 86M params | PASS | `tests/test_mae.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny ViT (patch 4, width 64, 2 blocks) MAE-pretrained 20 steps on synthetic 32×32 images, then a linear probe; finite outputs (final_recon_loss=1.543, linear_probe_top1=1, encoder_params=103104, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| ImageNet-1K pretraining and fine-tuning | NOT RUN | Requires a GPU cluster and ImageNet |

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
PYTHONPATH=src python -m mae_repro smoke --config configs/smoke.json --output-root results/smoke
```
