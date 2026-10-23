"""Equation-table generation, random splits, corruption, and tokenization."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from .operations import (
    Element,
    Equation,
    OPERATIONS,
    all_symbol_tokens,
    element_token,
    generate_equations,
)

EOS_TOKEN = "<eos>"
EQUALS_TOKEN = "="


class EquationTokenizer:
    """Atomic-symbol tokenizer matching the equation representation in the paper."""

    def __init__(self) -> None:
        symbols = list(all_symbol_tokens())
        operator_tokens = [op.operator_token for op in OPERATIONS.values()]
        ordered = [EOS_TOKEN, EQUALS_TOKEN, *operator_tokens, *symbols]
        if len(ordered) != len(set(ordered)):
            raise ValueError("Token inventory contains a duplicate symbol")
        self.tokens: tuple[str, ...] = tuple(ordered)
        self.stoi: dict[str, int] = {token: i for i, token in enumerate(self.tokens)}
        self.itos: tuple[str, ...] = self.tokens
        self.vocab_size = len(self.tokens)

    def encode_symbol(self, value: Element | str) -> int:
        token = value if isinstance(value, str) else element_token(value)
        try:
            return self.stoi[token]
        except KeyError as exc:
            raise KeyError(f"Unknown token {token!r}") from exc

    def encode_input(self, equation: Equation) -> list[int]:
        """Return ``[EOS, x, op, y, =]``; the model predicts the answer at '='."""
        return [
            self.stoi[EOS_TOKEN],
            self.encode_symbol(equation.left),
            self.stoi[equation.operator],
            self.encode_symbol(equation.right),
            self.stoi[EQUALS_TOKEN],
        ]

    def encode_target(self, answer: Element) -> int:
        return self.encode_symbol(answer)

    def decode(self, token_id: int) -> str:
        return self.itos[int(token_id)]

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens), "vocab_size": self.vocab_size}


@dataclass
class EncodedSplit:
    operation: str
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    val_inputs: torch.Tensor
    val_targets: torch.Tensor
    tokenizer: EquationTokenizer
    metadata: dict[str, Any]


def encode_equations(
    equations: list[Equation], tokenizer: EquationTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([tokenizer.encode_input(row) for row in equations], dtype=torch.long)
    targets = torch.tensor(
        [tokenizer.encode_target(row.answer) for row in equations], dtype=torch.long
    )
    return inputs, targets


def _corrupt_training_labels(
    equations: list[Equation], k: int, *, seed: int
) -> tuple[list[Equation], list[int]]:
    """Replace k training answers with answers from other sampled equations.

    The training split remains fully interpolable; validation labels are never
    changed.  As in the paper's description, a donor answer may coincide with
    the original answer by chance.
    """
    if k < 0:
        raise ValueError("k_outliers must be non-negative")
    if k == 0:
        return equations, []
    if k > len(equations):
        raise ValueError(
            f"Cannot inject {k} outliers into a training split of {len(equations)} rows"
        )
    if len(equations) < 2:
        raise ValueError("At least two training examples are required for outliers")

    rng = random.Random(seed)
    selected = rng.sample(range(len(equations)), k)
    donors = rng.sample(range(len(equations)), k)
    # Avoid using the same row as its own donor; this does not force a wrong
    # class because distinct equations can share the same correct answer.
    donors = [
        (donor + 1) % len(equations) if donor == target else donor
        for target, donor in zip(selected, donors)
    ]
    corrupted = list(equations)
    for target_idx, donor_idx in zip(selected, donors):
        target = corrupted[target_idx]
        donor = equations[donor_idx]
        corrupted[target_idx] = Equation(
            target.left, target.operator, target.right, donor.answer
        )
    return corrupted, selected


def make_split(
    operation: str,
    train_fraction: float,
    *,
    seed: int = 0,
    k_outliers: int = 0,
    tokenizer: EquationTokenizer | None = None,
) -> EncodedSplit:
    """Build a randomly sampled train/validation partition of the truth table.

    ``train_fraction`` is a proportion, e.g. ``0.5`` for the paper's 50/50
    modular-division experiment.  A given seed controls the table ordering,
    split, and outlier selection; training initialization is seeded separately.
    """
    if operation not in OPERATIONS:
        raise KeyError(f"Unknown operation {operation!r}; choose from {sorted(OPERATIONS)}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    op = OPERATIONS[operation]
    equations = generate_equations(op, shuffle=True, seed=seed)
    n_train = round(len(equations) * train_fraction)
    n_train = min(max(n_train, 1), len(equations) - 1)
    clean_train = equations[:n_train]
    val = equations[n_train:]
    train, outlier_indices = _corrupt_training_labels(
        clean_train, k_outliers, seed=seed + 17_071
    )
    tokenizer = tokenizer or EquationTokenizer()
    train_inputs, train_targets = encode_equations(train, tokenizer)
    val_inputs, val_targets = encode_equations(val, tokenizer)

    metadata = {
        "operation": operation,
        "paper_operation": op.paper_name,
        "formula": op.formula,
        "dataset_size": len(equations),
        "train_fraction_requested": train_fraction,
        "train_size": len(train),
        "validation_size": len(val),
        "train_fraction_actual": len(train) / len(equations),
        "seed": seed,
        "k_outliers_requested": k_outliers,
        "k_outliers_injected": len(outlier_indices),
        "outlier_train_indices": outlier_indices,
        "outlier_note": (
            "Answers are replaced from other randomly sampled training rows; "
            "validation labels remain clean."
            if k_outliers
            else None
        ),
    }
    return EncodedSplit(
        operation=operation,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        tokenizer=tokenizer,
        metadata=metadata,
    )
