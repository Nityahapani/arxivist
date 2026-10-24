# Verification log — arXiv:2305.14314

**Date:** 2026-10-23
**Status:** Implementation and smoke verification complete; paper-scale experiments not run.
**Author of this scaffold:** AI assistant (bot-authored commits). Results below were produced by actually running the listed commands in the authoring environment.

| Check | Result | Evidence |
|---|---|---|
| SIR JSON parses and conforms to Draft 7 schema | PASS — zero schema/format errors | `workspace/sir-registry/arxiv_2305_14314/sir.json` |
| NF4 levels match published table | PASS | `tests/test_qlora.py` |
| NF4 beats FP4/Int4 on Gaussian weights | PASS | `tests/test_qlora.py` |
| Double-quant round trip and bit overhead | PASS | `tests/test_qlora.py` |
| Adapters start at zero; base frozen | PASS | `tests/test_qlora.py` |
| All linear layers wrapped | PASS | `tests/test_qlora.py` |
| Pytest suite | PASS — 6 tests | `pytest -q` |
| CPU end-to-end smoke run | PASS — tiny GPT pretrained 5 steps, quantized to NF4 with double quantization, and LoRA-finetuned 6 steps on a new task; finite outputs (base_params=36672, trainable_params=6144, final_loss=2.794, is_paper_result=False) | Generated locally under `results/smoke/`; outputs are Git-ignored and omitted from this patch |
| Guanaco finetuning on OASST1 | NOT RUN | Requires LLaMA weights and GPU |
| MMLU / Vicuna evaluations | NOT RUN | Requires models and GPT-4 judging |

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
PYTHONPATH=src python -m qlora_repro smoke --config configs/smoke.json --output-root results/smoke
```
