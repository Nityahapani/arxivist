"""Training loop and CSV/checkpoint output for one paper experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import EncodedSplit, make_split
from .model import ModelConfig, TransformerLM
from .optim import make_optimizer, set_step_learning_rate


@dataclass
class TrainConfig:
    operation: str = "mod_divide"
    train_fraction: float = 0.5
    seed: int = 0
    steps: int = 100_000
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 1.0
    weight_decay_mode: str = "zero"
    update_noise_std: float = 0.0
    warmup_steps: int = 10
    batch_size: int | str | None = None
    full_batch: bool = False
    eval_interval: int = 100
    eval_batch_size: int = 2048
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    dropout: float = 0.0
    weight_noise: float = 0.0
    max_context_len: int = 50
    k_outliers: int = 0
    device: str = "auto"
    output_dir: str | None = None
    save_checkpoint: bool = False
    run_name: str = "run"
    variant: str = "default"

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0,1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.eval_interval < 1 or self.eval_batch_size < 1:
            raise ValueError("eval_interval and eval_batch_size must be positive")
        if self.batch_size not in (None, "full") and int(self.batch_size) < 1:
            raise ValueError("batch_size must be a positive integer, 'full', or None")
        if self.full_batch and self.batch_size not in (None, "full"):
            raise ValueError("Use either full_batch=True or a numeric batch_size")
        if self.k_outliers < 0:
            raise ValueError("k_outliers must be non-negative")
        ModelConfig(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            dropout=self.dropout,
            max_context_len=self.max_context_len,
            weight_noise=self.weight_noise,
        ).validate()


@dataclass
class TrainResult:
    model: TransformerLM
    metrics: list[dict[str, float | int]]
    summary: dict[str, Any]
    split: EncodedSplit
    device: torch.device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build has no CUDA device")
    return device


def _effective_batch_size(config: TrainConfig, n_train: int) -> int:
    if config.full_batch or config.batch_size == "full":
        return n_train
    if config.batch_size is not None:
        return min(int(config.batch_size), n_train)
    # Paper default: 512 examples or half of the training set, whichever is smaller.
    return min(512, max(1, math.ceil(n_train / 2)))


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int = 2048,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    n = len(targets)
    for start in range(0, n, batch_size):
        x = inputs[start : start + batch_size]
        y = targets[start : start + batch_size]
        logits = model.answer_logits(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
    if was_training:
        model.train()
    return total_loss / n, correct / n


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_model(config: TrainConfig) -> TrainResult:
    """Train one seed/split and optionally persist metrics and a final checkpoint."""
    config.validate()
    _seed_everything(config.seed)
    device = _select_device(config.device)
    split = make_split(
        config.operation,
        config.train_fraction,
        seed=config.seed,
        k_outliers=config.k_outliers,
    )

    model_config = ModelConfig(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        dropout=config.dropout,
        max_context_len=config.max_context_len,
        weight_noise=config.weight_noise,
    )
    model = TransformerLM(split.tokenizer.vocab_size, model_config).to(device)
    optimizer = make_optimizer(
        model,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        weight_decay_mode=config.weight_decay_mode,
        betas=(config.beta1, config.beta2),
        update_noise_std=config.update_noise_std,
    )

    x_train = split.train_inputs.to(device)
    y_train = split.train_targets.to(device)
    x_val = split.val_inputs.to(device)
    y_val = split.val_targets.to(device)
    batch_size = _effective_batch_size(config, len(y_train))
    rng = torch.Generator(device="cpu").manual_seed(config.seed + 91_337)
    permutation = torch.randperm(len(y_train), generator=rng)
    cursor = 0

    output_dir = Path(config.output_dir).expanduser() if config.output_dir else None
    metrics_fields = [
        "step", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "learning_rate", "elapsed_seconds",
    ]
    metrics_handle = None
    metrics_writer = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "run_config.json", asdict(config))
        _write_json(output_dir / "split_metadata.json", split.metadata)
        _write_json(output_dir / "tokenizer.json", split.tokenizer.to_dict())
        metrics_handle = (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metrics_fields)
        metrics_writer.writeheader()
        metrics_handle.flush()

    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    def record(step: int, learning_rate: float) -> None:
        train_loss, train_accuracy = evaluate(
            model, x_train, y_train, batch_size=config.eval_batch_size
        )
        val_loss, val_accuracy = evaluate(
            model, x_val, y_val, batch_size=config.eval_batch_size
        )
        history.append(
            {
                "step": step,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        if metrics_writer is not None and metrics_handle is not None:
            metrics_writer.writerow(history[-1])
            metrics_handle.flush()

    model.train()
    record(step=0, learning_rate=0.0)
    for step in range(1, config.steps + 1):
        if cursor >= len(y_train):
            permutation = torch.randperm(len(y_train), generator=rng)
            cursor = 0
        end = min(cursor + batch_size, len(y_train))
        indices = permutation[cursor:end].to(device)
        cursor = end
        if indices.numel() == 0:
            raise RuntimeError("Empty minibatch; check the dataset/sampling configuration")

        optimizer.zero_grad(set_to_none=True)
        logits = model.answer_logits(x_train.index_select(0, indices))
        loss = F.cross_entropy(logits, y_train.index_select(0, indices))
        loss.backward()
        lr_now = set_step_learning_rate(
            optimizer,
            base_lr=config.learning_rate,
            step_index=step - 1,
            warmup_steps=config.warmup_steps,
        )
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.steps:
            model.train()
            record(step=step, learning_rate=lr_now)
            model.train()

    if metrics_handle is not None:
        metrics_handle.close()

    best_val_accuracy = max(float(row["val_accuracy"]) for row in history)
    reached_99 = next(
        (int(row["step"]) for row in history if float(row["val_accuracy"]) >= 0.99),
        None,
    )
    final = history[-1]
    summary: dict[str, Any] = {
        "operation": config.operation,
        "train_fraction": config.train_fraction,
        "seed": config.seed,
        "steps_requested": config.steps,
        "steps_completed": int(final["step"]),
        "batch_size": batch_size,
        "train_size": len(y_train),
        "validation_size": len(y_val),
        "k_outliers": config.k_outliers,
        "best_validation_accuracy": best_val_accuracy,
        "final_train_accuracy": float(final["train_accuracy"]),
        "final_validation_accuracy": float(final["val_accuracy"]),
        "final_train_loss": float(final["train_loss"]),
        "final_validation_loss": float(final["val_loss"]),
        "first_logged_step_at_99pct_validation_accuracy": reached_99,
        "elapsed_seconds": float(final["elapsed_seconds"]),
        "device": str(device),
        "model_total_parameters": model.total_parameter_count(),
        "model_non_embedding_parameters": model.non_embedding_parameter_count(),
    }

    if output_dir is not None:
        _write_json(output_dir / "summary.json", summary)
        if config.save_checkpoint:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "vocab_size": split.tokenizer.vocab_size,
                    "tokenizer": split.tokenizer.to_dict(),
                    "run_config": asdict(config),
                    "summary": summary,
                },
                output_dir / "final.pt",
            )

    return TrainResult(model=model, metrics=history, summary=summary, split=split, device=device)
