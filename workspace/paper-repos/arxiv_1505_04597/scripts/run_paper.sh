#!/usr/bin/env bash
# Paper-scale protocol. Compute-intensive: requires the hardware listed in README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m unet_repro train \
  --config configs/paper.json \
  --output-root results/runs "$@"
