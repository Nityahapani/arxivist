# Verification log — arXiv:2106.09685

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2106_09685/sir.json` |
| Zero-initialised B reproduces the frozen base output | PASS | `tests/test_lora.py` |
| Merge/unmerge equivalence (layer and full model) | PASS | `tests/test_lora.py` |
| Gradients flow only into A and B | PASS | `tests/test_lora.py` |
| Trainable count equals 2·L̂·d·r for W_q/W_v injection | PASS | `tests/test_lora.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — 2-layer width-32 Transformer pretrained 60 steps on a synthetic task, then LoRA (r=4, W_q/W_v) adapted 60 steps on a second task; finite outputs (replaced_modules=4, total_params=26594, trainable_params=1090, trainable_fraction=0.04099) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| RoBERTa-base GLUE (Table 2) | NOT RUN | Requires GPU, `[hf]` extras, and network downloads; not available in authoring environment |
| GPT-2 / GPT-3 experiments | NOT RUN | Not implemented / out of scope |

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
PYTHONPATH=src python -m lora_repro smoke --config configs/smoke.json --output-root results/smoke
```
