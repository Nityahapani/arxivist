import torch
from torch import nn

from lora_repro.lora import LoRALinear, inject_lora, lora_param_count, lora_state_dict, \
    mark_only_lora_as_trainable, merge_all, unmerge_all
from lora_repro.model import TinyTransformer


def test_zero_init_matches_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 12)
    ref = base(torch.ones(3, 16)).detach().clone()
    layer = LoRALinear(base, r=4, alpha=8)
    assert torch.allclose(layer(torch.ones(3, 16)), ref)
    assert torch.count_nonzero(layer.lora_B) == 0
    assert layer.scaling == 2.0


def test_merge_unmerge_equivalence():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(10, 6), r=2, alpha=4)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(5, 10)
    y = layer(x)
    w0 = layer.base.weight.clone()
    layer.merge()
    assert torch.allclose(layer(x), y, atol=1e-6)
    layer.unmerge()
    assert torch.allclose(layer.base.weight, w0, atol=1e-6)
    assert torch.allclose(layer(x), y, atol=1e-6)


def test_only_lora_receives_gradients():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(8, 8), r=2)
    layer(torch.randn(4, 8)).sum().backward()
    assert layer.base.weight.grad is None
    assert layer.lora_B.grad is not None and layer.lora_A.grad is not None


def test_inject_counts_wq_wv_parameters():
    model = TinyTransformer(vocab_size=20, d_model=32, n_layers=3, n_heads=4, n_classes=2)
    replaced = inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=4)
    assert len(replaced) == 6
    mark_only_lora_as_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == lora_param_count(32, 3, 4, n_matrices=2)
    assert set(lora_state_dict(model)) == {n for n, _ in model.named_parameters() if "lora_" in n}


def test_merge_all_on_model():
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=20, d_model=16, n_layers=1, n_heads=2, n_classes=2).eval()
    inject_lora(model, ["q_proj", "v_proj"], r=2, alpha=2)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p)
    ids = torch.randint(0, 20, (3, 9))
    y = model(ids)
    merge_all(model)
    assert torch.allclose(model(ids), y, atol=1e-5)
    unmerge_all(model)
    assert torch.allclose(model(ids), y, atol=1e-5)
