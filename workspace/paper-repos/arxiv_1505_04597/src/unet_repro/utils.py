"""Shared experiment utilities: seeding, config loading, logging, parameter counts."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_device(pref: str | None = None) -> torch.device:
    if pref in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


class RunLogger:
    """Writes the resolved config, a metrics CSV + JSONL log, and a summary JSON."""

    def __init__(self, output_dir: str | Path, config: dict[str, Any]):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run_config.json").write_text(json.dumps(config, indent=2))
        self._jsonl = open(self.dir / "metrics.jsonl", "w", encoding="utf-8")
        self._csv_path = self.dir / "metrics.csv"
        self._fields: list[str] | None = None
        self.t0 = time.time()

    def log(self, row: dict[str, Any]) -> None:
        row = {"time_s": round(time.time() - self.t0, 3), **row}
        self._jsonl.write(json.dumps(row) + "\n")
        self._jsonl.flush()
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, self._fields).writeheader()
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, self._fields, extrasaction="ignore").writerow(row)

    def summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._jsonl.close()
        return summary
