import torch

from mae_repro.mae import CONFIGS, MaskedAutoencoder, patchify, random_masking, unpatchify
from mae_repro.vit import sincos_2d


def test_patchify_roundtrip():
    x = torch.rand(2, 3, 32, 32)
    p = patchify(x, 4)
    assert p.shape == (2, 64, 48)
    assert torch.equal(unpatchify(p, 4), x)


def test_random_masking_ratio_and_restore():
    x = torch.arange(2 * 16 * 1, dtype=torch.float).view(2, 16, 1)
    kept, mask, ids_restore = random_masking(x, 0.75, torch.Generator().manual_seed(0))
    assert kept.shape == (2, 4, 1)
    assert mask.sum(1).tolist() == [12.0, 12.0]
    # kept tokens are exactly the unmasked positions
    for b in range(2):
        assert sorted(kept[b, :, 0].tolist()) == sorted(x[b, mask[b] == 0, 0].tolist())


def test_loss_only_on_masked_patches():
    torch.manual_seed(0)
    m = MaskedAutoencoder(32, 3, "vit_tiny_patch4", 32, 1, 4, norm_pix_loss=False)
    imgs = torch.rand(2, 3, 32, 32)
    target = patchify(imgs, 4)
    mask = torch.zeros(2, 64); mask[:, :10] = 1
    pred = target.clone(); pred[:, 10:] += 5.0  # wrong only on visible patches
    assert m.loss(imgs, pred, mask).item() < 1e-10


def test_forward_shapes():
    m = MaskedAutoencoder(32, 3, "vit_tiny_patch4", 32, 1, 4)
    loss, pred, mask = m(torch.rand(2, 3, 32, 32))
    assert pred.shape == (2, 64, 48) and mask.shape == (2, 64) and loss.ndim == 0


def test_sincos_shape():
    assert sincos_2d(64, 8, True).shape == (65, 64)


def test_vit_base_encoder_size():
    m = MaskedAutoencoder(224, 3, "vit_base_patch16")
    enc = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("blocks", "patch_embed", "cls_token", "norm.")))
    assert 85.5e6 < enc < 86.2e6  # ViT-B/16 encoder ~86M
    assert CONFIGS["vit_large_patch16"]["depth"] == 24
