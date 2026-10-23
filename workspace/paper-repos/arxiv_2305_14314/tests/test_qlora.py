import torch

from qlora_repro.model import TinyGPT
from qlora_repro.qlora import NF4Linear, QLoRALinear, apply_qlora
from qlora_repro.quant import (bits_per_parameter, dequantize_blockwise, double_dequantize, double_quantize,
                               fp4_levels, int4_levels, nf4_levels, quantize_blockwise)

NF4_REFERENCE = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]


def test_nf4_levels_match_reference():
    lv = nf4_levels()
    assert lv.numel() == 16 and (lv == 0).sum() == 1
    assert torch.allclose(lv, torch.tensor(NF4_REFERENCE), atol=2e-4)


def test_nf4_beats_fp4_and_int4_on_normal_weights():
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    def mse(lv):
        c, a, s = quantize_blockwise(w, lv)
        return (dequantize_blockwise(c, a, s, lv) - w).pow(2).mean()
    assert mse(nf4_levels()) < mse(fp4_levels()) and mse(nf4_levels()) < mse(int4_levels())


def test_double_quantization_roundtrip_and_bits():
    absmax = torch.rand(1000) * 0.5 + 0.1
    rec = double_dequantize(*double_quantize(absmax))
    assert (rec - absmax).abs().max() < 2e-3
    assert abs(bits_per_parameter(double_quant=False) - 4.5) < 1e-9
    assert abs(bits_per_parameter() - (4 + 0.127)) < 1e-3


def test_qlora_linear_starts_at_quantized_base_and_freezes_base():
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64)
    ql = QLoRALinear(lin, r=4, alpha=8, dropout=0.0)
    x = torch.randn(3, 128)
    assert torch.allclose(ql(x), ql.base(x))
    rel = (ql.base.weight() - lin.weight).norm() / lin.weight.norm()
    assert rel < 0.2  # 4-bit reconstruction error
    assert not any(p.requires_grad for p in ql.base.parameters())


def test_apply_qlora_all_linear_layers():
    m = apply_qlora(TinyGPT(16, 64, 1, 2, 128, 8), r=4, alpha=8)
    n_q = sum(isinstance(x, QLoRALinear) for x in m.modules())
    assert n_q == 4  # qkv, proj, mlp.0, mlp.2 (head skipped)
    assert all("lora_" in n for n, p in m.named_parameters() if p.requires_grad)
