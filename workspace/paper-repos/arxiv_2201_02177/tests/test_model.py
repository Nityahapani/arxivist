import torch

from grokking_repro.data import EquationTokenizer
from grokking_repro.model import ModelConfig, TransformerLM


def test_paper_sized_model_output_and_parameter_count():
    tokenizer = EquationTokenizer()
    model = TransformerLM(tokenizer.vocab_size, ModelConfig())
    inputs = torch.randint(0, tokenizer.vocab_size, (4, 5))
    logits = model(inputs)
    assert logits.shape == (4, 5, tokenizer.vocab_size)
    assert 390_000 <= model.non_embedding_parameter_count() <= 410_000


def test_attention_is_causal():
    tokenizer = EquationTokenizer()
    torch.manual_seed(5)
    model = TransformerLM(
        tokenizer.vocab_size,
        ModelConfig(n_layers=1, n_heads=2, d_model=32, max_context_len=8),
    ).eval()
    first = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    changed_future = first.clone()
    changed_future[0, 4] = 5
    with torch.no_grad():
        a = model(first)
        b = model(changed_future)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)
    assert not torch.allclose(a[:, 4], b[:, 4])
