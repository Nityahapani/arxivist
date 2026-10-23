"""Small decoder-only Transformer used in the paper's algorithmic tasks.

The block layout follows the authors' released OpenAI implementation: sinusoidal
positions, causal attention, separate per-head Q/K/V projections, post-residual
LayerNorm, a 4x ReLU feed-forward block, and an untied linear unembedding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    dropout: float = 0.0
    max_context_len: int = 50
    weight_noise: float = 0.0
    ffn_multiplier: int = 4

    def validate(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        if self.d_model <= 0 or self.max_context_len <= 0:
            raise ValueError("d_model and max_context_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.weight_noise < 0:
            raise ValueError("weight_noise must be non-negative")


class NoisyLinear(nn.Linear):
    def __init__(self, *args, weight_noise: float = 0.0, **kwargs):
        self.weight_noise = float(weight_noise)
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.training and self.weight_noise:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            bias = self.bias
            if bias is not None:
                bias = bias + torch.randn_like(bias) * self.weight_noise
        else:
            weight, bias = self.weight, self.bias
        return F.linear(input, weight, bias)


class NoisyEmbedding(nn.Embedding):
    def __init__(self, *args, weight_noise: float = 0.0, **kwargs):
        self.weight_noise = float(weight_noise)
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        if self.training and self.weight_noise:
            weight = weight + torch.randn_like(weight) * self.weight_noise
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class NoisyLayerNorm(nn.LayerNorm):
    def __init__(self, *args, weight_noise: float = 0.0, **kwargs):
        self.weight_noise = float(weight_noise)
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        weight, bias = self.weight, self.bias
        if self.training and self.weight_noise:
            if weight is not None:
                weight = weight + torch.randn_like(weight) * self.weight_noise
            if bias is not None:
                bias = bias + torch.randn_like(bias) * self.weight_noise
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float):
        super().__init__()
        self.d_key = d_key
        self.query = NoisyLinear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.key = NoisyLinear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.value = NoisyLinear(d_model, d_key, bias=False, weight_noise=weight_noise)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_key)
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, weight_noise: float):
        super().__init__()
        d_key = d_model // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_key, weight_noise) for _ in range(n_heads)]
        )
        self.output = NoisyLinear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        joined = torch.cat([head(x, causal_mask) for head in self.heads], dim=-1)
        return self.output(joined)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, multiplier: int, weight_noise: float):
        super().__init__()
        hidden = multiplier * d_model
        self.up = NoisyLinear(d_model, hidden, bias=False, weight_noise=weight_noise)
        self.down = NoisyLinear(hidden, d_model, bias=False, weight_noise=weight_noise)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.relu(self.up(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadCausalAttention(
            config.d_model, config.n_heads, config.weight_noise
        )
        self.attention_norm = NoisyLayerNorm(
            config.d_model, weight_noise=config.weight_noise
        )
        self.feed_forward = FeedForward(
            config.d_model, config.ffn_multiplier, config.weight_noise
        )
        self.feed_forward_dropout = nn.Dropout(config.dropout)
        self.feed_forward_norm = NoisyLayerNorm(
            config.d_model, weight_noise=config.weight_noise
        )

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        x = self.attention_norm(x + self.attention(x, causal_mask))
        # The authors' implementation applies residual dropout to the FFN branch;
        # attention-output dropout is disabled there.
        x = self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))
        return x


def sinusoidal_positions(context_len: int, d_model: int) -> Tensor:
    positions = torch.arange(context_len, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    even = (dims % 2) == 0
    frequencies = torch.pow(10_000.0, torch.where(even, dims, dims - 1) / d_model)
    angles = positions / frequencies
    encodings = torch.where(even, torch.sin(angles), torch.cos(angles))
    return encodings


class TransformerLM(nn.Module):
    """Causal Transformer that predicts one answer token at each equation's '='."""

    def __init__(self, vocab_size: int, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.vocab_size = int(vocab_size)
        self.embedding = NoisyEmbedding(
            self.vocab_size, self.config.d_model, weight_noise=self.config.weight_noise
        )
        self.register_buffer(
            "position_encoding",
            sinusoidal_positions(self.config.max_context_len, self.config.d_model),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [DecoderBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.unembedding = NoisyLinear(
            self.config.d_model,
            self.vocab_size,
            bias=False,
            weight_noise=self.config.weight_noise,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected [batch, context] token IDs, got {tuple(input_ids.shape)}")
        _, context_len = input_ids.shape
        if context_len > self.config.max_context_len:
            raise ValueError(
                f"Context {context_len} exceeds max_context_len={self.config.max_context_len}"
            )
        x = self.embedding(input_ids)
        x = x + self.position_encoding[:context_len].to(dtype=x.dtype, device=x.device)
        causal_mask = torch.ones(
            context_len, context_len, dtype=torch.bool, device=x.device
        ).tril()
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.unembedding(x)

    def answer_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits from the final '=' position (the answer prediction)."""
        return self.forward(input_ids)[:, -1, :]

    def non_embedding_parameter_count(self) -> int:
        excluded = {"embedding.weight", "unembedding.weight"}
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name not in excluded
        )

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
