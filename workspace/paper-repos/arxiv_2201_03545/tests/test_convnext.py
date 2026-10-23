import torch
import numpy as np

from convnext_repro.augment import mixup_cutmix, one_hot_smooth
from convnext_repro.model import Block, ConvNeXt, drop_path


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_param_counts_match_paper():
    assert abs(n_params(ConvNeXt("convnext_tiny")) / 1e6 - 28.6) < 0.1   # Table 1: 28.6M
    assert abs(n_params(ConvNeXt("convnext_small")) / 1e6 - 50.2) < 0.1  # 50.2M


def test_stage_resolutions():
    m = ConvNeXt("convnext_atto_debug", n_classes=7)
    x = torch.rand(2, 3, 64, 64)
    shapes = []
    for ds, st in zip(m.downsample, m.stages):
        x = st(ds(x)); shapes.append(x.shape[-1])
    assert shapes == [16, 8, 4, 2]
    assert m(torch.rand(2, 3, 64, 64)).shape == (2, 7)


def test_layer_scale_makes_block_near_identity():
    b = Block(16, 0.0, 1e-6).eval()
    x = torch.randn(1, 16, 8, 8)
    assert (b(x) - x).abs().max() < 1e-4


def test_drop_path_eval_identity_and_train_scaling():
    x = torch.ones(1000, 4)
    assert torch.equal(drop_path(x, 0.5, False), x)
    y = drop_path(x, 0.5, True)
    assert set(y.unique().tolist()) <= {0.0, 2.0}


def test_mixup_targets_sum_to_one():
    x, y = torch.rand(8, 3, 16, 16), torch.randint(0, 5, (8,))
    for seed in range(4):
        xm, t = mixup_cutmix(x, y, 5, rng=np.random.default_rng(seed))
        assert xm.shape == x.shape and torch.allclose(t.sum(1), torch.ones(8))
    assert torch.allclose(one_hot_smooth(torch.tensor([1]), 4, 0.1), torch.tensor([[0.025, 0.925, 0.025, 0.025]]))
