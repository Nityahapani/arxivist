from trades_repro.train import run
from trades_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["robust_acc_pgd"] <= 1 and 0 <= s["natural_acc"] <= 1
