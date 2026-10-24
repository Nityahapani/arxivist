# Verification log — arXiv:2112.10752

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2112_10752/sir.json` |
| Autoencoder downsampling factor | PASS | `tests/test_ldm.py` |
| KL = 0 for standard normal | PASS | `tests/test_ldm.py` |
| VQ straight-through | PASS | `tests/test_ldm.py` |
| Schedule endpoints | PASS | `tests/test_ldm.py` |
| Cross-attention depends on context | PASS | `tests/test_ldm.py` |
| DDIM with oracle recovers x₀ | PASS | `tests/test_ldm.py` |
| CFG formula | PASS | `tests/test_ldm.py` |
| Pytest suite | PASS — 9 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny KL autoencoder (f=4) trained 4 steps, then a class-conditional latent UNet trained 4 steps and sampled with 4 DDIM steps and guidance; finite outputs (ae_params=196503, unet_params=918884, latent_scale=0.8935, downsampling_f=4) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| Unconditional / class-conditional / text-to-image benchmarks | NOT RUN | Requires datasets and GPUs |
| Inpainting and super-resolution | NOT RUN | Not configured |

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
PYTHONPATH=src python -m ldm_repro smoke --config configs/smoke.json --output-root results/smoke
```
