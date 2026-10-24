# Verification log — arXiv:2307.08621

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2307_08621/sir.json` |
| γ_i values | PASS | `tests/test_retnet.py` |
| Parallel = recurrent | PASS | `tests/test_retnet.py` |
| Chunkwise = parallel for several chunk sizes | PASS | `tests/test_retnet.py` |
| Recurrent decoding logits = parallel logits | PASS | `tests/test_retnet.py` |
| Constant-size state | PASS | `tests/test_retnet.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer RetNet trained 10 parallel-mode steps on synthetic Markov tokens, then recurrent vs parallel-recompute decoding timed; finite outputs (params=27136, final_loss=2.992, recurrent_tokens_per_s=1002, parallel_recompute_tokens_per_s=1131) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| LM scaling (1.3B–6.7B) | NOT RUN | Requires large corpus and GPU cluster |
| Inference cost at 7B / 8k | NOT RUN | Requires GPUs |

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
PYTHONPATH=src python -m retnet_repro smoke --config configs/smoke.json --output-root results/smoke
```
