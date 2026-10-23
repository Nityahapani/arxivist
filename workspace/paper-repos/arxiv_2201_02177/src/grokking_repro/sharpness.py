"""Keskar-style low-dimensional sharpness approximation used for Fig. 7."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import spearmanr

from .data import make_split
from .model import ModelConfig, TransformerLM


def _flatten_parameters(model: torch.nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()]).numpy().astype(np.float64)


def _assign_parameters(model: torch.nn.Module, flat: np.ndarray) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        values = torch.as_tensor(
            flat[offset : offset + count], dtype=parameter.dtype, device=parameter.device
        ).reshape_as(parameter)
        parameter.data.copy_(values)
        offset += count
    if offset != flat.size:
        raise ValueError(f"Parameter vector has {flat.size} values, expected {offset}")


def _loss_and_grad(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    model.eval()
    model.zero_grad(set_to_none=True)
    n = len(targets)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        x = inputs[start : start + batch_size]
        y = targets[start : start + batch_size]
        logits = model.answer_logits(x)
        summed_loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(summed_loss.detach().item())
        (summed_loss / n).backward()
    flat_grad = torch.cat(
        [p.grad.detach().reshape(-1).cpu() for p in model.parameters()]
    ).numpy().astype(np.float64)
    return total_loss / n, flat_grad


def keskar_sharpness(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int = 0,
    subspace_dim: int = 10,
    epsilon: float = 1e-3,
    max_iter: int = 10,
    batch_size: int = 512,
) -> dict[str, float | int]:
    """Approximate relative loss increase in a random low-dimensional subspace.

    The perturbation box follows Keskar et al. (2016), Sec. 2.2.2.  The objective
    is maximized with bounded L-BFGS-B; ``phi`` is the percentage increase
    ``100 * (max_loss - base_loss) / (1 + base_loss)``.
    """
    if subspace_dim < 1 or max_iter < 1 or epsilon <= 0:
        raise ValueError("subspace_dim, max_iter, and epsilon must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = next(model.parameters()).device
    x = inputs.to(device)
    y = targets.to(device)
    original_training_mode = model.training
    model.eval()
    x0 = _flatten_parameters(model)
    base_loss, _ = _loss_and_grad(model, x, y, batch_size=batch_size)

    if subspace_dim > x0.size:
        raise ValueError("subspace_dim cannot exceed the number of parameters")
    rng = np.random.default_rng(seed)
    directions = rng.uniform(-1.0, 1.0, size=(subspace_dim, x0.size))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, np.finfo(np.float64).tiny)
    basis = np.linalg.pinv(directions)
    radius = epsilon * (np.abs(directions @ x0) + 1.0)
    bounds = [(-float(r), float(r)) for r in radius]

    def objective(coords: np.ndarray) -> tuple[float, np.ndarray]:
        candidate = x0 + basis @ coords
        _assign_parameters(model, candidate)
        loss, grad_x = _loss_and_grad(model, x, y, batch_size=batch_size)
        grad_coords = basis.T @ grad_x
        return -loss, -grad_coords

    try:
        optimum, negative_max_loss, info = fmin_l_bfgs_b(
            objective,
            np.zeros(subspace_dim, dtype=np.float64),
            bounds=bounds,
            maxiter=max_iter,
            maxls=20,
            approx_grad=False,
        )
        max_loss = float(-negative_max_loss)
    finally:
        _assign_parameters(model, x0)
        model.train(original_training_mode)

    phi = 100.0 * (max_loss - base_loss) / (1.0 + base_loss)
    return {
        "base_loss": float(base_loss),
        "max_loss": max_loss,
        "phi": float(phi),
        "epsilon": float(epsilon),
        "subspace_dim": int(subspace_dim),
        "max_iter": int(max_iter),
        "seed": int(seed),
        "optimizer_iterations": int(info.get("nit", max_iter)),
    }


def load_checkpoint_for_sharpness(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[TransformerLM, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Restore a local run checkpoint and its exact train split."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_config = checkpoint["run_config"]
    split = make_split(
        run_config["operation"],
        float(run_config["train_fraction"]),
        seed=int(run_config["seed"]),
        k_outliers=int(run_config.get("k_outliers", 0)),
    )
    model = TransformerLM(
        int(checkpoint["vocab_size"]), ModelConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, split.train_inputs, split.train_targets, run_config


def spearman_summary(
    validation_accuracies: list[float], sharpness_values: list[float]
) -> dict[str, float | int]:
    if len(validation_accuracies) != len(sharpness_values) or len(validation_accuracies) < 3:
        raise ValueError("Provide at least three paired accuracy/sharpness observations")
    statistic = spearmanr(validation_accuracies, sharpness_values)
    return {
        "n": len(validation_accuracies),
        "spearman_rho": float(statistic.statistic),
        "p_value": float(statistic.pvalue),
    }


def write_sharpness_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
