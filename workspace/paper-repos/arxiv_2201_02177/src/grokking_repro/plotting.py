"""CSV aggregation and publication-style plots for Figures 1--7."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .model import ModelConfig, TransformerLM
from .operations import OPERATIONS, P, S5, element_token, s5_product


def _run_records(suite_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not suite_dir.exists():
        return records
    for run_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "run_config.json"
        if not (summary_path.exists() and metrics_path.exists() and config_path.exists()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        for row in metrics:
            for key in ("step",):
                row[key] = int(row[key])
            for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate", "elapsed_seconds"):
                row[key] = float(row[key])
        records.append({"dir": run_dir, "summary": summary, "config": config, "metrics": metrics})
    return records


def plot_figure1_division(suite_dir: str | Path, output_dir: str | Path) -> list[Path]:
    records = _run_records(Path(suite_dir))
    if not records:
        return []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    for metric_pair, filename, title, ylabel in (
        (("train_accuracy", "val_accuracy"), "figure1_division_accuracy.png", "Modular division mod 97: accuracy", "Exact-answer accuracy"),
        (("train_loss", "val_loss"), "figure4_division_loss.png", "Modular division mod 97: cross-entropy loss", "Cross-entropy loss"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for series, label, color in (
            (metric_pair[0], "Train", "#d55e00"),
            (metric_pair[1], "Validation", "#009e73"),
        ):
            arrays = []
            common_steps: np.ndarray | None = None
            for record in records:
                steps = np.asarray([row["step"] for row in record["metrics"]], dtype=float)
                values = np.asarray([row[series] for row in record["metrics"]], dtype=float)
                if common_steps is None:
                    common_steps = steps
                arrays.append(np.interp(common_steps, steps, values))
                ax.plot(steps[1:], values[1:], color=color, alpha=0.18, linewidth=0.8)
            if common_steps is not None and arrays:
                mean = np.mean(np.stack(arrays), axis=0)
                ax.plot(common_steps[1:], mean[1:], color=color, linewidth=2.4, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Optimization updates (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figure_paths.append(path)
    return figure_paths


def plot_figure1_s5_time(suite_dir: str | Path, output_dir: str | Path) -> Path | None:
    records = _run_records(Path(suite_dir))
    if not records:
        return None
    grouped: dict[float, list[float]] = {}
    for record in records:
        fraction = float(record["summary"]["train_fraction"])
        step = record["summary"].get("first_logged_step_at_99pct_validation_accuracy")
        if step is not None:
            grouped.setdefault(fraction, []).append(float(step))
    if not grouped:
        return None
    xs = sorted(grouped)
    medians = [float(np.median(grouped[x])) for x in xs]
    lows = [float(np.quantile(grouped[x], 0.25)) for x in xs]
    highs = [float(np.quantile(grouped[x], 0.75)) for x in xs]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "figure1_s5_time_to_99.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(np.asarray(xs) * 100, medians, marker="o", color="#0072b2")
    ax.fill_between(np.asarray(xs) * 100, lows, highs, color="#56b4e9", alpha=0.25, label="IQR")
    ax.set_yscale("log")
    ax.set_xlabel("Training equations (% of truth table)")
    ax.set_ylabel("Updates to first logged validation accuracy ≥ 99% (log scale)")
    ax.set_title("S5 product: data fraction vs. time to generalization")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_peak_accuracy(
    suite_dir: Path,
    output_path: Path,
    *,
    group_key: str,
    title: str,
    xlabel: str,
    series_label: str,
) -> Path | None:
    records = _run_records(suite_dir)
    if not records:
        return None
    grouped: dict[str, dict[float, list[float]]] = {}
    for record in records:
        summary = record["summary"]
        key = str(summary.get(group_key, record["config"].get(group_key, "default")))
        fraction = float(summary["train_fraction"])
        peak = max(float(row["val_accuracy"]) for row in record["metrics"])
        grouped.setdefault(key, {}).setdefault(fraction, []).append(peak)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for label, points in sorted(grouped.items()):
        xs = sorted(points)
        means = [float(np.mean(points[x])) * 100 for x in xs]
        stds = [float(np.std(points[x], ddof=1)) * 100 if len(points[x]) > 1 else 0.0 for x in xs]
        ax.errorbar(
            np.asarray(xs) * 100, means, yerr=stds, marker="o", capsize=2,
            linewidth=1.4, label=f"{series_label}: {label}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best validation exact-answer accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_figure2_variety(suite_dir: str | Path, output_dir: str | Path) -> Path | None:
    return _plot_peak_accuracy(
        Path(suite_dir), Path(output_dir) / "figure2_variety.png",
        group_key="operation", title="Generalization across the paper's 12 operations",
        xlabel="Training equations (% of truth table)", series_label="Operation",
    )


def plot_figure2_ablations(suite_dir: str | Path, output_dir: str | Path) -> Path | None:
    return _plot_peak_accuracy(
        Path(suite_dir), Path(output_dir) / "figure2_ablations.png",
        group_key="variant", title="S5 product: optimization and regularization ablations",
        xlabel="Training equations (% of truth table)", series_label="Variant",
    )


def plot_figure6_outliers(suite_dir: str | Path, output_dir: str | Path) -> Path | None:
    records = _run_records(Path(suite_dir))
    if not records:
        return None
    grouped: dict[int, dict[float, list[float]]] = {}
    for record in records:
        summary = record["summary"]
        k = int(summary.get("k_outliers", 0))
        fraction = float(summary["train_fraction"])
        peak = max(float(row["val_accuracy"]) for row in record["metrics"])
        grouped.setdefault(k, {}).setdefault(fraction, []).append(peak)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "figure6_outliers.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for k, points in sorted(grouped.items()):
        xs = sorted(points)
        means = [float(np.mean(points[x])) * 100 for x in xs]
        ax.plot(np.asarray(xs) * 100, means, marker="o", linewidth=1.5, label=f"k={k}")
    ax.set_xlabel("Training equations (% of truth table)")
    ax.set_ylabel("Best validation exact-answer accuracy (%)")
    ax.set_title("Label outliers and data efficiency")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _generated_subgroup() -> tuple[tuple[int, int, int, int, int], ...]:
    g1 = (3, 4, 2, 0, 1)  # (0,3)(1,4)
    g2 = (0, 2, 1, 4, 3)  # (1,2)(3,4)
    identity = tuple(range(5))
    closure = {identity, g1, g2}
    changed = True
    while changed:
        changed = False
        for left in tuple(closure):
            for right in tuple(closure):
                result = s5_product(left, right)
                if result not in closure:
                    closure.add(result)
                    changed = True
    return tuple(sorted(closure))


def _coset_labels(elements: tuple[tuple[int, ...], ...]) -> np.ndarray:
    subgroup = _generated_subgroup()
    labels: dict[tuple[int, ...], int] = {}
    coset_id = 0
    for representative in elements:
        if representative in labels:
            continue
        for member in subgroup:
            labels[s5_product(representative, member)] = coset_id
        coset_id += 1
    return np.asarray([labels[element] for element in elements])


def _load_unembedding(checkpoint_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = TransformerLM(int(checkpoint["vocab_size"]), model_config)
    model.load_state_dict(checkpoint["state_dict"])
    tokens = checkpoint["tokenizer"]["tokens"]
    token_ids = {token: i for i, token in enumerate(tokens)}
    return model.unembedding.weight.detach().cpu().numpy(), token_ids


def plot_figure3_embeddings(
    modular_addition_checkpoint: str | Path,
    s5_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 0,
) -> list[Path]:
    """t-SNE of learned unembedding rows for addition and S5 product."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    matrix, token_ids = _load_unembedding(Path(modular_addition_checkpoint))
    residues = [str(i) for i in range(P)]
    indices = np.asarray([token_ids[token] for token in residues])
    points = TSNE(
        n_components=2, init="pca", learning_rate="auto", random_state=random_state,
        perplexity=30, max_iter=1000,
    ).fit_transform(matrix[indices])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = np.asarray([i % 8 for i in range(P)])
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, cmap="tab10", s=24, alpha=0.85)
    point_index = {residue: i for i, residue in enumerate(range(P))}
    for residue in range(P):
        other = (residue + 8) % P
        a, b = points[point_index[residue]], points[point_index[other]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#777777", alpha=0.12, linewidth=0.5)
    fig.colorbar(scatter, ax=ax, label="Residue modulo 8")
    ax.set_title("Modular addition: t-SNE of output-layer symbol weights")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    path = output / "figure3_addition_unembedding_tsne.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    matrix, token_ids = _load_unembedding(Path(s5_checkpoint))
    elements = tuple(S5)
    tokens = [element_token(item) for item in elements]
    indices = np.asarray([token_ids[token] for token in tokens])
    points = TSNE(
        n_components=2, init="pca", learning_rate="auto", random_state=random_state,
        perplexity=30, max_iter=1000,
    ).fit_transform(matrix[indices])
    labels = _coset_labels(elements)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=32, alpha=0.9)
    fig.colorbar(scatter, ax=ax, label="Coset of the paper-noted subgroup (chosen convention)")
    ax.set_title("S5 product: t-SNE of output-layer symbol weights")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    path = output / "figure3_s5_unembedding_tsne.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def export_operation_table(
    operation: str,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    """Write a complete numeric/symbolic operation table to CSV."""
    import csv
    from .operations import generate_equations

    op = OPERATIONS[operation]
    equations = generate_equations(op, shuffle=False)
    rows = list(op.elements)
    columns = list(op.elements)
    if operation == "mod_divide":
        columns = [value for value in columns if int(value) != 0]
    if limit is not None:
        rows = rows[:limit]
        columns = columns[:limit]
    lookup = {(row.left, row.right): row.answer for row in equations}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"x \\ {operation}", *[element_token(x) for x in columns]])
        for left in rows:
            writer.writerow([element_token(left), *[element_token(lookup[(left, right)]) for right in columns]])
    return path


def plot_sharpness_csv(csv_path: str | Path, output_dir: str | Path) -> Path | None:
    path = Path(csv_path)
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        return None
    accuracy = np.asarray([float(row["validation_accuracy"]) for row in rows]) * 100
    phi = np.asarray([float(row["phi"]) for row in rows])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    out = output / "figure7_accuracy_by_sharpness.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(phi, accuracy, alpha=0.8, color="#0072b2")
    ax.set_xlabel("Keskar-style sharpness approximation φ (%)")
    ax.set_ylabel("Validation exact-answer accuracy (%)")
    ax.set_title("Validation accuracy vs. local sharpness (S5 product)")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_all(results_root: str | Path, output_dir: str | Path) -> list[Path]:
    root, output = Path(results_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for path in plot_figure1_division(root / "figure1_modular_division", output):
        generated.append(path)
    for path in (
        plot_figure1_s5_time(root / "figure1_s5_learning_time", output),
        plot_figure2_variety(root / "figure2_variety", output),
        plot_figure2_ablations(root / "figure2_ablations", output),
        plot_figure6_outliers(root / "appendix_a4_outliers", output),
        plot_sharpness_csv(root / "appendix_a5_sharpness" / "sharpness.csv", output),
    ):
        if path is not None:
            generated.append(path)
    return generated
