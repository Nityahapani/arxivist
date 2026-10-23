import torch

from grokking_repro.model import NoisyLinear
from grokking_repro.optim import PaperAdam


def test_decoupled_decay_modes_match_zero_and_initialization_targets():
    toward_zero = torch.nn.Parameter(torch.tensor([2.0]))
    zero_opt = PaperAdam(
        [toward_zero], lr=0.1, betas=(0.0, 0.0), weight_decay=0.5,
        weight_decay_mode="zero",
    )
    toward_zero.grad = torch.zeros_like(toward_zero)
    zero_opt.step()
    assert torch.allclose(toward_zero, torch.tensor([1.9]))

    toward_init = torch.nn.Parameter(torch.tensor([2.0]))
    init_opt = PaperAdam(
        [toward_init], lr=0.1, betas=(0.0, 0.0), weight_decay=0.5,
        weight_decay_mode="init",
    )
    toward_init.grad = torch.zeros_like(toward_init)
    init_opt.step()  # records the initialization
    with torch.no_grad():
        toward_init.fill_(0.0)
    toward_init.grad = torch.zeros_like(toward_init)
    init_opt.step()
    assert torch.allclose(toward_init, torch.tensor([0.1]))


def test_gaussian_update_noise_is_applied_and_linear_weight_noise_is_eval_off():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = PaperAdam(
        [parameter], lr=0.1, betas=(0.0, 0.0), update_noise_std=1.0
    )
    parameter.grad = torch.zeros_like(parameter)
    torch.manual_seed(0)
    optimizer.step()
    assert parameter.item() != 0.0

    layer = NoisyLinear(3, 2, bias=False, weight_noise=0.1)
    inputs = torch.ones(1, 3)
    layer.train()
    first, second = layer(inputs), layer(inputs)
    assert not torch.allclose(first, second)
    layer.eval()
    first, second = layer(inputs), layer(inputs)
    assert torch.allclose(first, second)
