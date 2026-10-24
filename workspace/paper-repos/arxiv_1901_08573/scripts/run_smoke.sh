#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m trades_repro smoke \
  --config configs/smoke.json \
  --output-root results/smoke
