import math

import torch

from dinov2_repro.heads import DINOHead
from dinov2_repro.losses import dino_loss, ibot_loss, koleo_loss, sinkhorn_knopp
from dinov2_repro.masking import sample_masks
from dinov2_repro.model import DinoV2
from dinov2_repro.schedules import cosine, teacher_temp_at
from dinov2_repro.vit import DinoViT


def test_sinkhorn_marginals():
    torch.manual_seed(0)
    Q = sinkhorn_knopp(torch.randn(64, 16), 0.1, n_iters=50)
    assert torch.allclose(Q.sum(1), torch.ones(64), atol=1e-4)
    assert torch.allclose(Q.sum(0), torch.full((16,), 64 / 16), atol=5e-2)  # equipartition across prototypes
    raw = torch.softmax(torch.randn(64, 16) / 0.1, -1).sum(0)
    assert Q.sum(0).std() < raw.std()


def test_koleo_prefers_spread_features():
    spread = torch.eye(8)
    clustered = torch.eye(8) * 0.01 + torch.tensor([1.0] + [0] * 7)
    assert koleo_loss(spread) < koleo_loss(clustered)


def test_dino_loss_skips_same_view_and_is_minimal_at_match():
    t = [torch.softmax(torch.randn(4, 10) / 0.05, -1) for _ in range(2)]
    s_good = [torch.log(t[1]) * 0.1, torch.log(t[0]) * 0.1]  # student crop i predicts the *other* teacher crop
    s_bad = [torch.randn(4, 10), torch.randn(4, 10)]
    assert dino_loss(s_good, t) < dino_loss(s_bad, t)


def test_ibot_loss_masked_positions_only():
    s = torch.randn(2, 5, 7, requires_grad=True)
    masks = torch.tensor([[1, 0, 0, 1, 0], [0, 0, 1, 0, 0]], dtype=torch.bool)
    tp = torch.softmax(torch.randn(3, 7), -1)
    ibot_loss(s, tp, masks).backward()
    assert torch.equal(s.grad.abs().sum(-1) > 0, masks)


def test_vit_s14_params_and_pos_interpolation():
    m = DinoViT.from_name("vit_s14")
    n = sum(p.numel() for p in m.parameters())
    assert 21.0e6 < n < 22.5e6  # ViT-S/14 ~21M
    small = DinoViT(32, 8, 16, 1, 2)
    cls, patches = small(torch.rand(2, 3, 16, 16))
    assert cls.shape == (2, 16) and patches.shape == (2, 4, 16)


def test_mask_token_replaces_patches():
    torch.manual_seed(0)
    m = DinoViT(16, 8, 16, 1, 2).eval()
    x = torch.rand(1, 3, 16, 16)
    masks = torch.tensor([[True, True, True, True]])
    a = m(x, masks)[0]
    b = m(torch.rand(1, 3, 16, 16), masks)[0]
    assert torch.allclose(a, b, atol=1e-6)  # fully masked input -> image content irrelevant


def test_teacher_ema_and_schedules():
    model = DinoV2(DinoViT(16, 8, 16, 1, 2), 8, 8, 16, 8)
    assert all(not p.requires_grad for p in model.teacher.parameters())
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.teacher.parameters()]
    model.update_teacher(0.9)
    for pt, b, ps in zip(model.teacher.parameters(), before, model.student.parameters()):
        assert torch.allclose(pt, 0.9 * b + 0.1 * ps)
    assert math.isclose(cosine(0, 10, 0.04, 0.2), 0.04) and math.isclose(cosine(10, 10, 0.04, 0.2), 0.2)
    assert teacher_temp_at(0, 10) == 0.04 and teacher_temp_at(10, 10) == 0.07


def test_masks_ratio():
    g = torch.Generator().manual_seed(0)
    m = sample_masks(200, 100, g)
    frac = m.float().mean(1)
    assert ((frac == 0) | ((frac >= 0.1) & (frac <= 0.5))).all() and 0.3 < (frac > 0).float().mean() < 0.7
