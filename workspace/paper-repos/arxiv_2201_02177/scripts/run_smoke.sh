#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m grokking_repro run \
  --config configs/smoke.json \
  --suite smoke \
  --output-root results/smoke
