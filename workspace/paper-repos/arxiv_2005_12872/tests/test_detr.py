import itertools

import torch

from detr_repro.box_ops import cxcywh_to_xyxy, generalized_box_iou, xyxy_to_cxcywh
from detr_repro.criterion import SetCriterion
from detr_repro.matcher import hungarian_match
from detr_repro.model import DETR, sine_position_encoding


def test_giou_properties():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.allclose(generalized_box_iou(a, a), torch.ones(1, 1))
    far = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    assert generalized_box_iou(a, far).item() < 0
    b = torch.rand(5, 4)
    assert torch.allclose(xyxy_to_cxcywh(cxcywh_to_xyxy(b)), b, atol=1e-6)


def test_matcher_is_optimal():
    torch.manual_seed(0)
    logits, boxes = torch.randn(1, 5, 4), torch.rand(1, 5, 4) * 0.5 + 0.25
    t = {"labels": torch.tensor([0, 2, 1]), "boxes": torch.rand(3, 4) * 0.5 + 0.25}
    i, j = hungarian_match(logits, boxes, [t])[0]
    prob = logits.softmax(-1)[0]
    def cost(pi, tj):
        return (-prob[pi, t["labels"][tj]] + 5 * (boxes[0, pi] - t["boxes"][tj]).abs().sum()
                - 2 * generalized_box_iou(cxcywh_to_xyxy(boxes[0, pi:pi + 1]), cxcywh_to_xyxy(t["boxes"][tj:tj + 1]))[0, 0]).item()
    best = min(sum(cost(p, k) for k, p in enumerate(perm)) for perm in itertools.permutations(range(5), 3))
    assert abs(sum(cost(int(p), int(k)) for p, k in zip(i, j)) - best) < 1e-5


def test_loss_invariant_to_target_order():
    torch.manual_seed(0)
    out = {"logits": torch.randn(2, 1, 6, 4), "boxes": torch.rand(2, 1, 6, 4) * 0.5 + 0.25}
    t = {"labels": torch.tensor([0, 1, 2]), "boxes": torch.rand(3, 4) * 0.5 + 0.25}
    perm = torch.tensor([2, 0, 1])
    crit = SetCriterion(3)
    a, _ = crit(out, [t])
    b, _ = crit(out, [{"labels": t["labels"][perm], "boxes": t["boxes"][perm]}])
    assert torch.allclose(a, b)


def test_no_object_weight():
    w = SetCriterion(5).empty_weight
    assert torch.allclose(w, torch.tensor([1, 1, 1, 1, 1, 0.1]))


def test_position_encoding_and_outputs():
    assert sine_position_encoding(4, 6, 32).shape == (24, 32)
    m = DETR(3, 7, 32, 4, 1, 2, 64, 0.0, "resnet18", 8).eval()
    out = m(torch.rand(2, 3, 64, 64))
    assert out["logits"].shape == (2, 2, 7, 4) and out["boxes"].shape == (2, 2, 7, 4)
    assert (out["boxes"] >= 0).all() and (out["boxes"] <= 1).all()


def test_detr_r50_parameter_count():
    n = sum(p.numel() for p in DETR(91).parameters())
    assert 40.5e6 < n < 42.0e6  # paper: 41M
