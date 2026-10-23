import json
from pathlib import Path

import torch

from grokking_repro.optim import warmup_learning_rate
from grokking_repro.train import TrainConfig, train_model


def test_warmup_schedule_is_linear_then_constant():
    assert warmup_learning_rate(1e-3, 0, 10) == 1e-4
    assert warmup_learning_rate(1e-3, 9, 10) == 1e-3
    assert warmup_learning_rate(1e-3, 10, 10) == 1e-3


def test_short_train_writes_reproducible_metrics_and_checkpoint(tmp_path: Path):
    out = tmp_path / "smoke"
    config = TrainConfig(
        operation="mod_add",
        train_fraction=0.5,
        seed=2,
        steps=4,
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=1.0,
        warmup_steps=2,
        batch_size=64,
        eval_interval=2,
        eval_batch_size=512,
        n_layers=1,
        n_heads=2,
        d_model=32,
        max_context_len=50,
        device="cpu",
        output_dir=str(out),
        save_checkpoint=True,
    )
    result = train_model(config)
    assert len(result.metrics) == 3  # initial evaluation plus steps 2 and 4
    assert result.summary["steps_completed"] == 4
    assert result.summary["model_non_embedding_parameters"] < 20_000
    for name in ("run_config.json", "split_metadata.json", "tokenizer.json", "metrics.csv", "summary.json", "final.pt"):
        assert (out / name).exists(), name
    assert json.loads((out / "summary.json").read_text())["operation"] == "mod_add"
    checkpoint = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["run_config"]["seed"] == 2
