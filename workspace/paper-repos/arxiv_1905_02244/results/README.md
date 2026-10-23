# Reproduction outputs

Metrics, checkpoints, and figures are written here when the commands in the top-level README are run.

- `runs/` contains one directory per configuration/seed with the resolved config, a metrics CSV/JSONL log, and a summary JSON.
- `smoke/` is reserved for the short pipeline check; smoke outputs are **not** paper results.

Paper-scale runs and checkpoints are git-ignored. This repository currently contains **no paper-scale results**: no synthetic or fabricated metrics have been added, and full-scale runs were not performed in the authoring environment (CPU-only). Results are to be filled in by the contributor who runs the experiments listed in `comparison/benchmark_comparison.md`.
