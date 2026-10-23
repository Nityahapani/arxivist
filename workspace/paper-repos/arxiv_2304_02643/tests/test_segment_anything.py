import torch

from segment_anything_repro.image_encoder import ImageEncoderViT, equally_spaced_global, window_partition, window_unpartition
from segment_anything_repro.losses import dice_loss, focal_loss, sam_loss
from segment_anything_repro.model import SAM
from segment_anything_repro.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from segment_anything_repro.prompts import corrective_point, noisy_box


def tiny():
    torch.manual_seed(0)
    return SAM(64, 8, 32, 4, 4, 4, None, 32, 2, 4, 64).eval()


def test_window_partition_roundtrip_with_padding():
    x = torch.randn(2, 10, 9, 5)
    w, pad = window_partition(x, 4)
    assert w.shape == (2 * 3 * 3, 4, 4, 5) and pad == (12, 12)
    assert torch.equal(window_unpartition(w, 4, pad, (10, 9)), x)


def test_global_blocks_equally_spaced():
    assert equally_spaced_global(32) == (7, 15, 23, 31)
    enc = ImageEncoderViT(64, 8, 32, 8, 4, 4, None, 16)
    assert [b.ws for b in enc.blocks] == [4, 0, 4, 0, 4, 0, 4, 0]
    assert enc(torch.randn(1, 3, 64, 64)).shape == (1, 16, 8, 8)


def test_random_fourier_pe():
    pe = PositionEmbeddingRandom(8)
    g = pe.grid(4, 6)
    assert g.shape == (16, 4, 6) and g.abs().max() <= 1.0


def test_prompt_encoder_padding_and_shapes():
    enc = PromptEncoder(32, (8, 8), (64, 64))
    pts, lbl = torch.tensor([[[10.0, 20.0]]]), torch.ones(1, 1, dtype=torch.long)
    s, d = enc((pts, lbl))
    assert s.shape == (1, 2, 32) and d.shape == (1, 32, 8, 8)  # padded with not-a-point
    s2, _ = enc((pts, lbl), torch.tensor([[1.0, 2.0, 30.0, 40.0]]))
    assert s2.shape == (1, 3, 32)  # no padding when a box is present
    _, d3 = enc(masks=torch.randn(1, 1, 32, 32))
    assert d3.shape == (1, 32, 8, 8)
    s_neg, _ = enc((pts, torch.zeros(1, 1, dtype=torch.long)))
    assert not torch.allclose(s_neg[:, 0], s[:, 0])


def test_decoder_multimask_and_single():
    m = tiny()
    emb = m.image_encoder(torch.randn(1, 3, 64, 64))
    pts = (torch.tensor([[[30.0, 30.0]]]), torch.ones(1, 1, dtype=torch.long))
    low, iou = m.predict(emb, pts, multimask=True)
    assert low.shape == (1, 3, 32, 32) and iou.shape == (1, 3)  # 4x upscaled embedding
    low1, iou1 = m.predict(emb, pts, multimask=False)
    assert low1.shape == (1, 1, 32, 32) and iou1.shape == (1, 1)
    assert m.upscale(low1).shape == (1, 1, 64, 64)


def test_focal_and_dice():
    t = torch.zeros(1, 4, 4); t[0, :2] = 1
    perfect = (t * 2 - 1) * 50
    assert dice_loss(perfect, t).item() < 1e-3 and focal_loss(perfect, t).item() < 1e-6
    logits = torch.zeros(1, 4, 4)
    ce = torch.log(torch.tensor(2.0))
    expected = (0.25 * 0.5 + 0.75 * 0.5) * ce * 0.25  # alpha mix * CE * (1-p_t)^2 with p=0.5
    assert torch.allclose(focal_loss(logits, t), expected.view(1), atol=1e-6)


def test_min_loss_over_masks():
    t = torch.zeros(1, 4, 4); t[0, :2] = 1
    good, bad = (t * 2 - 1) * 10, -(t * 2 - 1) * 10
    masks = torch.stack([bad[0], good[0], bad[0]])[None]
    loss, best = sam_loss(masks, torch.zeros(1, 3), t)
    assert best.item() == 1
    single, _ = sam_loss(good[None], torch.ones(1, 1), t)
    # only the best mask's seg loss counts; IoU MSE covers all 3 masks (targets 0, 1, 0 vs predictions 0)
    assert abs(loss.item() - (single.item() + 1 / 3)) < 1e-4


def test_corrective_point_in_error_region():
    g = torch.Generator().manual_seed(0)
    gt = torch.zeros(16, 16, dtype=torch.bool); gt[4:12, 4:12] = True
    pred = torch.zeros_like(gt); pred[4:12, 4:8] = True
    for _ in range(10):
        p, lab = corrective_point(pred, gt, g)
        x, y = p.long()
        assert gt[y, x] and not pred[y, x] and lab == 1
    assert corrective_point(gt, gt, g) == (None, None)
    b = noisy_box(gt, g)
    assert b.shape == (4,) and (b >= 0).all()
