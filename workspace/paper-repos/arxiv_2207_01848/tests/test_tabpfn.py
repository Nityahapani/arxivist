import torch

from tabpfn_repro.model import TabPFN, pfn_attention_mask, predict_proba
from tabpfn_repro.prior import pad_features, sample_mlp_dataset, sample_prior_batch


def small():
    torch.manual_seed(0)
    return TabPFN(8, 3, 16, 2, 2, 32).eval()


def test_attention_mask():
    m = pfn_attention_mask(3, 5)
    assert m[:, :3].all() and not m[:, 3:].any()


def test_test_predictions_independent_of_other_test_points():
    model = small()
    xtr, ytr = torch.randn(1, 6, 8), torch.randint(0, 3, (1, 6))
    xte = torch.randn(1, 4, 8)
    a = model(xtr, ytr, xte)
    xte2 = xte.clone(); xte2[:, 1:] = torch.randn(1, 3, 8)
    b = model(xtr, ytr, xte2)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)


def test_invariant_to_training_set_order():
    model = small()
    xtr, ytr, xte = torch.randn(1, 6, 8), torch.randint(0, 3, (1, 6)), torch.randn(1, 4, 8)
    perm = torch.randperm(6)
    assert torch.allclose(model(xtr, ytr, xte), model(xtr[:, perm], ytr[:, perm], xte), atol=1e-5)


def test_prior_samples():
    g = torch.Generator().manual_seed(0)
    X, y = sample_mlp_dataset(200, 5, 3, g, scm=True)
    assert X.shape == (200, 5) and y.max() < 3 and torch.isfinite(X).all()
    Xb, yb, d, c = sample_prior_batch(4, 50, 10, 4, g)
    assert Xb.shape == (4, 50, 10) and (Xb[..., d:] == 0).all() and yb.max() < c


def test_feature_padding_rescales():
    X = torch.ones(2, 3, 4)
    P = pad_features(X, 8)
    assert P.shape == (2, 3, 8) and torch.allclose(P[..., :4], torch.full((2, 3, 4), 2.0))


def test_predict_proba_sums_to_one():
    model = small()
    p = predict_proba(model, torch.randn(10, 5), torch.randint(0, 3, (10,)), torch.randn(7, 5), 3, 3, None, 8)
    assert p.shape == (7, 3) and torch.allclose(p.sum(1), torch.ones(7), atol=1e-5)
