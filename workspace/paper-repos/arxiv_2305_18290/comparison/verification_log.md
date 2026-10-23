# Verification log — arXiv:2305.18290

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2305_18290/sir.json` |
| Loss = log 2 at initialisation | PASS | `tests/test_dpo.py` |
| Gradient equals −β·σ(r̂_l − r̂_w) | PASS | `tests/test_dpo.py` |
| Prompt tokens masked from log-probs | PASS | `tests/test_dpo.py` |
| Expected-loss minimiser equals RLHF optimum | PASS | `tests/test_dpo.py` |
| Pytest suite | PASS — 5 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny GPT SFT'd for 5 steps, then trained 10 DPO steps on 256 synthetic Bradley-Terry preference pairs; finite outputs (params=10016, sft_true_reward=0.2715, dpo_true_reward=0.2969, kl_to_ref=0.03425) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| IMDb sentiment with GPT-2-large | NOT RUN | Requires HF model, classifier, GPU |
| TL;DR and Anthropic HH with GPT-4 judging | NOT RUN | Requires large models and API access |

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
PYTHONPATH=src python -m dpo_repro smoke --config configs/smoke.json --output-root results/smoke
```
