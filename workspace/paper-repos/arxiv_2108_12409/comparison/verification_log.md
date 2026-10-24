# Verification log — arXiv:2108.12409

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2108_12409/sir.json` |
| Slopes match the geometric rule (8, 16, 12 heads) | PASS | `tests/test_alibi.py` |
| Bias matrix values / zero diagonal | PASS | `tests/test_alibi.py` |
| Model runs beyond training length; strictly causal | PASS | `tests/test_alibi.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer width-32 LMs (ALiBi and sinusoidal) trained 40 steps at L=32 on a synthetic Markov stream, evaluated at L=32/64/128; finite outputs (L_train=32, params=18176, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| WikiText-103 extrapolation experiments | NOT RUN | Requires multi-GPU training |

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
PYTHONPATH=src python -m alibi_repro smoke --config configs/smoke.json --output-root results/smoke
```
