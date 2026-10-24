# Verification log — arXiv:2405.21060

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2405_21060/sir.json` |
| segsum | PASS | `tests/test_ssd.py` |
| Recurrent = quadratic | PASS | `tests/test_ssd.py` |
| Chunked = recurrent incl. final state | PASS | `tests/test_ssd.py` |
| Initial-state support | PASS | `tests/test_ssd.py` |
| Block step decoding = chunked forward | PASS | `tests/test_ssd.py` |
| Pytest suite | PASS — 7 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer Mamba-2 LM trained 10 steps on MQAR (4 key-value pairs, 4 queries); finite outputs (params=17848, final_loss=20.81, mqar_accuracy=0.008789, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| LM scaling on the Pile | NOT RUN | Requires GPUs and corpus |
| Kernel speed benchmarks | NOT RUN | Requires GPU / Triton |

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
PYTHONPATH=src python -m mamba2_repro smoke --config configs/smoke.json --output-root results/smoke
```
