import torch

from deeponet_repro.model import DeepONet, FNNBaseline
from deeponet_repro.train import run
from deeponet_repro.utils import load_config


def test_unstacked_and_stacked_shapes():
    u, y = torch.randn(7, 100), torch.rand(7, 1)
    for stacked in (False, True):
        net = DeepONet(100, p=8, stacked=stacked)
        assert net(u, y).shape == (7, 1)
        assert net.branch_out(u).shape == (7, 8)
    assert len(DeepONet(100, p=8, stacked=True).branch) == 8


def test_output_is_inner_product_plus_bias():
    net = DeepONet(10, p=4)
    with torch.no_grad():
        net.b0.fill_(0.5)
    u, y = torch.randn(3, 10), torch.rand(3, 1)
    expected = (net.branch_out(u) * net.trunk(y)).sum(-1, keepdim=True) + 0.5
    assert torch.allclose(net(u, y), expected)


def test_fnn_baseline_shape():
    assert FNNBaseline(10)(torch.randn(2, 10), torch.rand(2, 1)).shape == (2, 1)


def test_training_reduces_loss(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 200
    cfg["train"]["eval_every"] = 20
    s = run(cfg, tmp_path, smoke=True)
    r = s["runs"]["unstacked"]
    first = float(open(next(tmp_path.rglob("metrics.csv"))).read().splitlines()[1].split(",")[3])
    assert r["final_test_mse"] < first
    assert r["final_test_mse"] < 0.5
