"""Synthetic language-modelling data: a random first-order Markov chain over the vocabulary."""
import torch


def markov_tokens(n_tokens: int, vocab: int, g: torch.Generator, temperature: float = 0.5) -> torch.Tensor:
    trans = torch.softmax(torch.randn(vocab, vocab, generator=g) / temperature, -1)
    out = torch.empty(n_tokens, dtype=torch.long)
    out[0] = 0
    for i in range(1, n_tokens):
        out[i] = torch.multinomial(trans[out[i - 1]], 1, generator=g)
    return out


def batches(tokens, seq_len, batch_size, g):
    idx = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), generator=g)
    x = torch.stack([tokens[i:i + seq_len + 1] for i in idx])
    return x[:, :-1], x[:, 1:]
