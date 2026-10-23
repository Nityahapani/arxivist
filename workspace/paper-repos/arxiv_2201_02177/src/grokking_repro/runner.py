"""Manifest-driven experiment-grid runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterator

from .train import TrainConfig, train_model


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "defaults" not in payload or "suites" not in payload:
        raise ValueError("Manifest must contain 'defaults' and 'suites'")
    return payload


def _slug_fraction(value: float) -> str:
    return f"{value:.4g}".replace(".", "p")


def _make_run_specs(defaults: dict[str, Any], suite: dict[str, Any]) -> Iterator[dict[str, Any]]:
    default_fractions = defaults.get("train_fractions", [0.5])
    default_seeds = defaults.get("seeds", [0])
    default_outliers = [0]

    if "runs" in suite:
        for fixed_run in suite["runs"]:
            yield deepcopy(fixed_run)
        return

    operations = suite.get("operations")
    if operations is None:
        operations = [suite["operation"]]
    fractions = suite.get("train_fractions", default_fractions)
    seeds = suite.get("seeds", default_seeds)
    outliers = suite.get("k_outliers_values", default_outliers)
    variants = suite.get("variants", [{"name": "default"}])

    for operation in operations:
        for fraction in fractions:
            for seed in seeds:
                for k_outliers in outliers:
                    for variant in variants:
                        yield {
                            "operation": operation,
                            "train_fraction": fraction,
                            "seed": seed,
                            "k_outliers": k_outliers,
                            "variant": variant.get("name", "default"),
                            "variant_config": variant,
                        }


def _build_train_config(
    defaults: dict[str, Any],
    suite: dict[str, Any],
    spec: dict[str, Any],
    *,
    output_dir: Path,
    device: str | None,
    steps_override: int | None,
) -> TrainConfig:
    training = deepcopy(defaults.get("training", {}))
    training.update(suite.get("training", {}))
    model = deepcopy(defaults.get("model", {}))
    model.update(suite.get("model", {}))
    variant = spec.pop("variant_config", {})
    training.update(variant.get("training", {}))
    model.update(variant.get("model", {}))

    if steps_override is not None:
        training["steps"] = steps_override
    run_device = device or suite.get("device") or defaults.get("device", "auto")
    save_checkpoint = bool(suite.get("save_checkpoint", False))
    if "save_checkpoint" in variant:
        save_checkpoint = bool(variant["save_checkpoint"])

    return TrainConfig(
        operation=spec["operation"],
        train_fraction=float(spec["train_fraction"]),
        seed=int(spec["seed"]),
        k_outliers=int(spec.get("k_outliers", 0)),
        output_dir=str(output_dir),
        run_name=str(spec.get("name", suite.get("name", "run"))),
        variant=str(spec.get("variant", "default")),
        device=str(run_device),
        save_checkpoint=save_checkpoint,
        **training,
        **model,
    )


def run_suite(
    manifest: dict[str, Any],
    suite_name: str,
    *,
    output_root: str | Path,
    device: str | None = None,
    steps_override: int | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Execute one manifest suite. ``limit`` is useful for smoke runs."""
    if suite_name not in manifest["suites"]:
        raise KeyError(f"Unknown suite {suite_name!r}; available: {sorted(manifest['suites'])}")
    root = Path(output_root)
    suite = manifest["suites"][suite_name]
    defaults = manifest["defaults"]
    results: list[dict[str, Any]] = []

    for index, spec in enumerate(_make_run_specs(defaults, suite)):
        if limit is not None and index >= limit:
            break
        op = spec["operation"]
        fraction = float(spec["train_fraction"])
        seed = int(spec["seed"])
        variant_name = spec.get("variant", "default")
        k_outliers = int(spec.get("k_outliers", 0))
        run_id = (
            f"{op}_frac-{_slug_fraction(fraction)}_seed-{seed}_"
            f"{variant_name}_k-{k_outliers}"
        )
        run_dir = root / suite_name / run_id
        config = _build_train_config(
            defaults,
            suite,
            spec,
            output_dir=run_dir,
            device=device,
            steps_override=steps_override,
        )
        if verbose:
            print(
                f"[{index + 1}] {suite_name}: {op}, fraction={fraction:g}, "
                f"seed={seed}, variant={variant_name}, k={k_outliers}, "
                f"steps={config.steps}, device={config.device}",
                flush=True,
            )
        result = train_model(config)
        result.summary["suite"] = suite_name
        result.summary["run_id"] = run_id
        # train_model writes its summary before the suite/run labels are appended.
        (run_dir / "summary.json").write_text(
            json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(result.summary)
        if verbose:
            print(json.dumps(result.summary, sort_keys=True), flush=True)
        del result
    return results


def run_suites(
    manifest: dict[str, Any],
    suite_names: list[str] | None,
    *,
    output_root: str | Path,
    device: str | None = None,
    steps_override: int | None = None,
    limit_per_suite: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    names = suite_names or list(manifest["suites"])
    return {
        name: run_suite(
            manifest,
            name,
            output_root=output_root,
            device=device,
            steps_override=steps_override,
            limit=limit_per_suite,
        )
        for name in names
    }
