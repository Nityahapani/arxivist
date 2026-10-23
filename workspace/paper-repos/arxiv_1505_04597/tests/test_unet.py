import numpy as np
import torch

from unet_repro.augment import elastic_deform
from unet_repro.data import synthetic_cells
from unet_repro.infer import overlap_tile_predict
from unet_repro.model import UNet, center_crop, output_size
from unet_repro.weights import unet_weight_map


def test_output_size_572_to_388():
    assert output_size(572) == 388


def test_forward_valid_convolutions():
    m = UNet(1, 2, base=4, depth=4)
    assert m(torch.rand(1, 1, 572, 572)).shape == (1, 2, 388, 388)


def test_full_unet_parameter_count():
    n = sum(p.numel() for p in UNet().parameters())
    assert 31.0e6 < n < 31.1e6  # ~31.03M for the original configuration


def test_center_crop():
    x = torch.arange(36).view(1, 1, 6, 6)
    assert torch.equal(center_crop(x, 2, 2)[0, 0], torch.tensor([[14, 15], [20, 21]]))


def test_weight_map_peaks_between_touching_cells():
    inst = np.zeros((20, 30), dtype=int)
    inst[5:15, 2:14] = 1
    inst[5:15, 16:28] = 2  # 2-pixel gap at columns 14-15
    w = unet_weight_map(inst, w0=10, sigma=5)
    assert w[10, 14] > w[10, 0] and w[10, 14] > 5


def test_elastic_deform_preserves_shape_and_labels():
    img, lab, inst = synthetic_cells(64, 5, np.random.default_rng(0))
    out, inst2 = elastic_deform(img, inst, rng=np.random.default_rng(1))
    assert out.shape == img.shape and set(np.unique(inst2)) <= set(np.unique(inst))


def test_overlap_tile_covers_image():
    m = UNet(1, 2, base=4, depth=2)
    img = np.random.rand(1, 50, 70).astype(np.float32)
    assert overlap_tile_predict(m, img, 92, depth=2).shape == (2, 50, 70)
