"""Command-line interface for training, analysis, and paper figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

from .data import make_split
from .plotting import (
    export_operation_table,
    plot_all,
    plot_figure3_embeddings,
)
from .runner import load_manifest, run_suite, run_suites
from .sharpness import (
    keskar_sharpness,
    load_checkpoint_for_sharpness,
    spearman_summary,
)
from .train import evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grokking-repro",
        description="Reproduce the experiments in Power et al. (2022), arXiv:2201.02177",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one manifest suite")
    run.add_argument("--config", default="configs/experiment_suite.json")
    run.add_argument("--suite", required=True)
    run.add_argument("--output-root", default="results/runs")
    run.add_argument("--device", default=None, help="auto, cpu, or cuda[:N]")
    run.add_argument("--steps", type=int, default=None, help="override steps for smoke tests")
    run.add_argument("--limit", type=int, default=None, help="run at most N jobs from this suite")

    run_all = subparsers.add_parser("run-all", help="run every suite in the manifest")
    run_all.add_argument("--config", default="configs/experiment_suite.json")
    run_all.add_argument("--output-root", default="results/runs")
    run_all.add_argument("--device", default=None)
    run_all.add_argument("--steps", type=int, default=None, help="override steps for smoke tests")
    run_all.add_argument("--limit-per-suite", type=int, default=None)
    run_all.add_argument("--suites", nargs="*", default=None)

    plot = subparsers.add_parser("plot", help="aggregate runs and create available figures")
    plot.add_argument("--results-root", default="results/runs")
    plot.add_argument("--output-dir", default="results/figures")
    plot.add_argument("--include-figure3", action="store_true")

    table = subparsers.add_parser("table", help="export one complete operation truth table as CSV")
    table.add_argument("--operation", required=True)
    table.add_argument("--output", required=True)
    table.add_argument("--limit", type=int, default=None, help="export only the first N rows/columns")

    sharpness = subparsers.add_parser("sharpness", help="compute appendix A.5 sharpness on checkpoints")
    sharpness.add_argument("--checkpoints", required=True, help="directory containing final.pt checkpoints")
    sharpness.add_argument("--output-csv", default="results/runs/appendix_a5_sharpness/sharpness.csv")
    sharpness.add_argument("--output-json", default="results/runs/appendix_a5_sharpness/sharpness_summary.json")
    sharpness.add_argument("--device", default="cpu")
    sharpness.add_argument("--subspace-dim", type=int, default=10)
    sharpness.add_argument("--epsilon", type=float, default=1e-3)
    sharpness.add_argument("--max-iter", type=int, default=10)
    sharpness.add_argument("--batch-size", type=int, default=512)

    return parser


def _find_checkpoint(suite_dir: Path, operation: str) -> Path | None:
    for candidate in sorted(suite_dir.rglob("final.pt")):
        try:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
            if checkpoint.get("run_config", {}).get("operation") == operation:
                return candidate
        except Exception:
            continue
    return None


def _run_sharpness(args: argparse.Namespace) -> int:
    checkpoint_root = Path(args.checkpoints)
    checkpoints = sorted(checkpoint_root.rglob("final.pt")) if checkpoint_root.is_dir() else [checkpoint_root]
    if not checkpoints or not checkpoints[0].exists():
        raise FileNotFoundError(f"No final.pt checkpoints found at {checkpoint_root}")

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        print(f"Sharpness: {checkpoint_path}", flush=True)
        model, train_x, train_y, config = load_checkpoint_for_sharpness(
            checkpoint_path, device=args.device
        )
        model.eval()
        split = make_split(
            config["operation"],
            float(config["train_fraction"]),
            seed=int(config["seed"]),
            k_outliers=int(config.get("k_outliers", 0)),
        )
        val_loss, val_accuracy = evaluate(
            model,
            split.val_inputs.to(args.device),
            split.val_targets.to(args.device),
            batch_size=args.batch_size,
        )
        result = keskar_sharpness(
            model,
            train_x,
            train_y,
            seed=int(config["seed"]),
            subspace_dim=args.subspace_dim,
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
        )
        row: dict[str, object] = {
            "checkpoint": str(checkpoint_path),
            "operation": config["operation"],
            "train_fraction": float(config["train_fraction"]),
            "seed": int(config["seed"]),
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            **result,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del model

    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"n": len(rows), "observations": rows}
    if len(rows) >= 3:
        summary["correlation"] = spearman_summary(
            [float(row["validation_accuracy"]) for row in rows],
            [float(row["phi"]) for row in rows],
        )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        manifest = load_manifest(args.config)
        run_suite(
            manifest,
            args.suite,
            output_root=args.output_root,
            device=args.device,
            steps_override=args.steps,
            limit=args.limit,
        )
        return 0
    if args.command == "run-all":
        manifest = load_manifest(args.config)
        run_suites(
            manifest,
            args.suites,
            output_root=args.output_root,
            device=args.device,
            steps_override=args.steps,
            limit_per_suite=args.limit_per_suite,
        )
        return 0
    if args.command == "plot":
        generated = plot_all(args.results_root, args.output_dir)
        if args.include_figure3:
            root = Path(args.results_root) / "figure3_embeddings"
            addition = _find_checkpoint(root, "mod_add")
            s5 = _find_checkpoint(root, "s5_product")
            if addition is not None and s5 is not None:
                generated.extend(plot_figure3_embeddings(addition, s5, args.output_dir))
            else:
                print("Skipping Figure 3: both modular-addition and S5 final.pt checkpoints are required.")
        if generated:
            print("Generated figures:")
            for path in generated:
                print(f"  {path}")
        else:
            print("No suite results found yet; run the relevant suites first.")
        return 0
    if args.command == "table":
        print(export_operation_table(args.operation, args.output, limit=args.limit))
        return 0
    if args.command == "sharpness":
        return _run_sharpness(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
