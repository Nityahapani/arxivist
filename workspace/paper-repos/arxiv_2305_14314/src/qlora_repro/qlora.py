"""QLoRA linear layer (Section 3, Eq. 5-6):

Y^BF16 = X^BF16 doubleDequant(c1^FP32, c2^k-bit, W^NF4) + X^BF16 L1^BF16 L2^BF16

The base weight is stored as NF4 codes with double-quantized constants and is frozen; only the LoRA
adapters L1, L2 receive gradients. Computation dtype is bfloat16 in the paper; float32 is used on CPU.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .quant import dequantize_blockwise, double_dequantize, double_quantize, nf4_levels, quantize_blockwise


class NF4Linear(nn.Module):
    def __init__(self, linear: nn.Linear, block=64, dq_block=256, double_quant=True, compute_dtype=torch.float32):
        super().__init__()
        self.block, self.dq_block, self.double_quant, self.compute_dtype = block, dq_block, double_quant, compute_dtype
        self.register_buffer("levels", nf4_levels(), persistent=False)
        codes, absmax, shape = quantize_blockwise(linear.weight, self.levels, block)
        self.shape = shape
        self.register_buffer("codes", codes)
        if double_quant:
            q, scale, mean, n = double_quantize(absmax, dq_block)
            self.register_buffer("c2", q); self.register_buffer("c1", scale); self.register_buffer("c_mean", mean)
            self.n_blocks = n
        else:
            self.register_buffer("absmax", absmax)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def weight(self):
        absmax = double_dequantize(self.c2, self.c1, self.c_mean, self.n_blocks, self.dq_block) if self.double_quant else self.absmax
        return dequantize_blockwise(self.codes, absmax, self.shape, self.levels, self.block).to(self.compute_dtype)

    def forward(self, x):
        return F.linear(x.to(self.compute_dtype), self.weight(), None if self.bias is None else self.bias.to(self.compute_dtype))


class QLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=64, alpha=16, dropout=0.1, **qkw):
        super().__init__()
        self.base = NF4Linear(linear, **qkw)
        self.lora_A = nn.Parameter(torch.empty(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def apply_qlora(model: nn.Module, r=64, alpha=16, dropout=0.1, skip=("head",), **qkw) -> nn.Module:
    """Replace every nn.Linear (all layers, Section 4 "LoRA on all linear layers") except `skip`."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name not in skip:
            setattr(model, name, QLoRALinear(child, r, alpha, dropout, **qkw))
        else:
            apply_qlora(child, r, alpha, dropout, skip, **qkw)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    return model
