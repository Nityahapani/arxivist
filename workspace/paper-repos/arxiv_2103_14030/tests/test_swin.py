import torch

from swin_repro.model import (SwinBlock, SwinTransformer, WindowAttention, relative_position_index,
                              shifted_window_mask, window_partition, window_reverse)


def test_partition_roundtrip():
    x = torch.randn(2, 14, 14, 8)
    w = window_partition(x, 7)
    assert w.shape == (8, 49, 8)
    assert torch.equal(window_reverse(w, 7, 14, 14), x)


def test_relative_index_range():
    idx = relative_position_index(7)
    assert idx.shape == (49, 49) and idx.min() == 0 and idx.max() == 13 * 13 - 1
    assert torch.all(idx.diagonal() == idx[0, 0])  # zero offset maps to the same entry


def test_shift_mask_structure():
    mask = shifted_window_mask(8, 8, 4, 2)
    assert mask.shape == (4, 16, 16)
    assert torch.all(mask[0] == 0)  # top-left window is a single region
    assert (mask[-1] < 0).any()     # bottom-right window mixes 4 regions


def test_shifted_attention_does_not_mix_regions():
    torch.manual_seed(0)
    blk = SwinBlock(8, 2, 4, 2, (8, 8)).eval()
    x = torch.randn(1, 64, 8)
    y = x.clone().view(1, 8, 8, 8)
    # perturb a token in the bottom-right corner; after cyclic shift it shares a window with
    # top-left tokens but must not influence them
    y[0, 7, 7] += 10.0
    out_a, out_b = blk(x), blk(y.view(1, 64, 8))
    assert torch.allclose(out_a.view(1, 8, 8, 8)[0, 0, 0], out_b.view(1, 8, 8, 8)[0, 0, 0], atol=1e-5)


def test_swin_tiny_parameter_count():
    n = sum(p.numel() for p in SwinTransformer("swin_tiny").parameters())
    assert abs(n / 1e6 - 28.3) < 0.1  # Table 1: 28M (28.3M)


def test_debug_forward():
    m = SwinTransformer("swin_debug", img_size=64, n_classes=5)
    assert m(torch.rand(2, 3, 64, 64)).shape == (2, 5)
