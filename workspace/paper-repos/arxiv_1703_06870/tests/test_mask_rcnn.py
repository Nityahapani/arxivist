import torch

from mask_rcnn_repro.box_ops import decode, encode, nms
from mask_rcnn_repro.data import synthetic_instances
from mask_rcnn_repro.model import MaskRCNN
from mask_rcnn_repro.roi_align import fpn_level, roi_align
from mask_rcnn_repro.roi_heads import mask_loss


def test_roi_align_exact_on_linear_feature():
    """For f(y, x) = 2y + 3x + 1 (in image units with cell centres at i + 0.5), RoIAlign returns the value at
    each bin's centre (bilinear interpolation is exact for linear functions; average of symmetric samples)."""
    H = W = 16
    yy, xx = torch.meshgrid(torch.arange(H).float() + 0.5, torch.arange(W).float() + 0.5, indexing="ij")
    f = (2 * yy + 3 * xx + 1)[None, None]
    roi = torch.tensor([[0, 2.3, 3.1, 10.7, 9.9]])
    out = roi_align(f, roi, 4, 1.0, 2)[0, 0]
    x0, y0, x1, y1 = 2.3, 3.1, 10.7, 9.9
    cy = y0 + (torch.arange(4) + 0.5) * (y1 - y0) / 4
    cx = x0 + (torch.arange(4) + 0.5) * (x1 - x0) / 4
    assert torch.allclose(out, 2 * cy[:, None] + 3 * cx[None] + 1, atol=1e-4)


def test_roi_align_no_quantization():
    f = torch.randn(1, 1, 16, 16)
    a = roi_align(f, torch.tensor([[0, 2.0, 2.0, 9.0, 9.0]]), 7, 1.0)
    b = roi_align(f, torch.tensor([[0, 2.2, 2.0, 9.2, 9.0]]), 7, 1.0)
    assert not torch.allclose(a, b)  # sub-pixel shifts change the output (RoIPool would not)


def test_fpn_level_assignment():
    rois = torch.tensor([[0, 0, 0, 224, 224], [0, 0, 0, 112, 112], [0, 0, 0, 448, 448], [0, 0, 0, 10, 10]]).float()
    assert fpn_level(rois).tolist() == [4, 3, 5, 2]


def test_box_coding_roundtrip_and_nms():
    ref = torch.tensor([[0.0, 0, 10, 10], [5, 5, 20, 30]])
    gt = torch.tensor([[1.0, 2, 12, 9], [4, 6, 25, 28]])
    assert torch.allclose(decode(ref, encode(ref, gt, (10, 10, 5, 5)), (10, 10, 5, 5)), gt, atol=1e-5)
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]])
    assert nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5).tolist() == [0, 2]


def test_mask_loss_only_uses_gt_class():
    logits = torch.randn(3, 4, 28, 28, requires_grad=True)
    labels = torch.tensor([2, 0, 3])
    mask_loss(logits, labels, torch.rand(3, 28, 28).round()).backward()
    nz = logits.grad.abs().sum((2, 3)) > 0
    assert nz.tolist() == [[False, False, True, False], [True, False, False, False], [False, False, False, True]]


def test_model_train_and_inference():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    m = MaskRCNN(3, "resnet18", 8, 16, (8, 16, 24, 32, 48), 28, 100, 30, 16, 32, 8)
    imgs, targets = synthetic_instances(2, 64, 3, g)
    losses = m(imgs, targets, g)
    assert set(losses) == {"rpn_cls", "rpn_box", "cls", "box", "mask"}
    sum(losses.values()).backward()
    m.eval()
    out = m(imgs)
    assert len(out) == 2
    for r in out:
        assert r["masks"].shape == (len(r["boxes"]), 28, 28)


def test_r50_fpn_parameter_count():
    n = sum(p.numel() for p in MaskRCNN(80).parameters())
    assert 43.5e6 < n < 45.0e6  # ~44.4M for R-50-FPN Mask R-CNN (80 classes)
