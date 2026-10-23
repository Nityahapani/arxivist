"""Command-line entry point: `smoke` (tiny pipeline check) and `train` (paper protocol)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import run
from .utils import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dinov2-repro", description="Independent reproduction of DINOv2 (arXiv:2304.07193)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("smoke", "tiny end-to-end pipeline check (not a paper result)"),
                            ("train", "run the configured paper-scale protocol")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True)
        p.add_argument("--output-root", required=True)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--steps", type=int, default=None, help="debug override; truncates the protocol")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    if args.steps is not None:
        cfg.setdefault("train", {})["steps"] = args.steps
        cfg["truncated_by_override"] = True
    summary = run(cfg, Path(args.output_root), smoke=args.cmd == "smoke")
    print(json.dumps(summary, indent=2))
    return 0
