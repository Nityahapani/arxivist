import torch

from grokking_repro.data import EOS_TOKEN, EQUALS_TOKEN, EquationTokenizer, make_split
from grokking_repro.operations import OPERATIONS


def test_tokenizer_uses_atomic_symbols_and_full_shared_vocabulary():
    tokenizer = EquationTokenizer()
    assert tokenizer.vocab_size == 2 + len(OPERATIONS) + 97 + 120
    assert tokenizer.decode(tokenizer.stoi[EOS_TOKEN]) == EOS_TOKEN
    assert tokenizer.decode(tokenizer.stoi[EQUALS_TOKEN]) == EQUALS_TOKEN
    assert tokenizer.encode_symbol(96) != tokenizer.encode_symbol(9)
    assert tokenizer.encode_symbol((0, 1, 2, 3, 4)) == tokenizer.stoi["01234"]


def test_split_is_deterministic_exhaustive_and_disjoint():
    first = make_split("mod_add", 0.5, seed=13)
    second = make_split("mod_add", 0.5, seed=13)
    assert torch.equal(first.train_inputs, second.train_inputs)
    assert torch.equal(first.train_targets, second.train_targets)
    assert len(first.train_targets) + len(first.val_targets) == 97**2
    train_rows = {tuple(row.tolist()) for row in first.train_inputs}
    val_rows = {tuple(row.tolist()) for row in first.val_inputs}
    assert train_rows.isdisjoint(val_rows)
    assert first.metadata["train_size"] == round(97**2 * 0.5)


def test_division_split_excludes_zero_divisor():
    split = make_split("mod_divide", 0.5, seed=3)
    divisor_id = split.tokenizer.stoi["0"]
    assert not torch.any(split.train_inputs[:, 3] == divisor_id)
    assert not torch.any(split.val_inputs[:, 3] == divisor_id)
    assert len(split.train_targets) + len(split.val_targets) == 97 * 96


def test_outliers_are_training_only_and_counted():
    clean = make_split("mod_add", 0.5, seed=7, k_outliers=0)
    noisy = make_split("mod_add", 0.5, seed=7, k_outliers=5)
    assert noisy.metadata["k_outliers_injected"] == 5
    assert noisy.metadata["outlier_train_indices"]
    assert torch.equal(clean.val_inputs, noisy.val_inputs)
    assert torch.equal(clean.val_targets, noisy.val_targets)
    assert len(clean.train_targets) == len(noisy.train_targets)
    # The deterministic donor assignment changes at least one target for this seed.
    assert not torch.equal(clean.train_targets, noisy.train_targets)
