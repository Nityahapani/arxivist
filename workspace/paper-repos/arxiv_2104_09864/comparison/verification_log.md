# Verification log — arXiv:2104.09864

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2104_09864/sir.json` |
| Matrix, complex, and efficient forms agree | PASS | `tests/test_rope.py` |
| Scores depend only on m − n | PASS | `tests/test_rope.py` |
| Norm preservation and θ_i values | PASS | `tests/test_rope.py` |
| Long-term decay bound decreases | PASS | `tests/test_rope.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer RoFormer and sinusoidal baseline trained 10 MLM steps on synthetic periodic sequences; finite outputs (is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| English MLM + GLUE | NOT RUN | Requires corpora and GPUs |
| Enwik8 Performer + RoPE | NOT RUN | Requires data |
| Chinese pretraining + CAIL2019-SCM | NOT RUN | Requires data |

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
PYTHONPATH=src python -m roformer_repro smoke --config configs/smoke.json --output-root results/smoke
```
