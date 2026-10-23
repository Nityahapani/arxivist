import math

import torch

from grokking_repro.data import EquationTokenizer
from grokking_repro.model import ModelConfig, TransformerLM
from grokking_repro.sharpness import keskar_sharpness


def test_keskar_sharpness_returns_finite_phi_and_restores_parameters():
    torch.manual_seed(4)
    tokenizer = EquationTokenizer()
    model = TransformerLM(
        tokenizer.vocab_size,
        ModelConfig(n_layers=1, n_heads=2, d_model=32, max_context_len=8),
    )
    inputs = torch.randint(0, tokenizer.vocab_size, (12, 5))
    targets = torch.randint(0, tokenizer.vocab_size, (12,))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = keskar_sharpness(
        model,
        inputs,
        targets,
        seed=3,
        subspace_dim=2,
        epsilon=1e-3,
        max_iter=1,
        batch_size=6,
    )
    assert math.isfinite(result["phi"])
    assert result["subspace_dim"] == 2
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
