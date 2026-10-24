# Verification log — arXiv:2205.14135

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2205_14135/sir.json` |
| Forward and log-sum-exp exact (causal / non-causal, uneven blocks) | PASS | `tests/test_flash.py` |
| Gradients exact | PASS | `tests/test_flash.py` |
| Block-sparse = masked dense | PASS | `tests/test_flash.py` |
| Block sizes and IO scaling | PASS | `tests/test_flash.py` |
| Pytest suite | PASS — 11 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 1-layer GPT trained 6 steps with FlashAttention and with standard attention from identical initialisation; losses compared; finite outputs (params=11680, loss_abs_diff=0, hbm_access_ratio_standard_over_flash=1.998, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| Wall-clock benchmarks (BERT, GPT-2, LRA) | NOT RUN | Requires CUDA kernel and A100s |
| Path-X / Path-256 | NOT RUN | Requires long-context GPU training |

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
PYTHONPATH=src python -m flashattention_repro smoke --config configs/smoke.json --output-root results/smoke
```
