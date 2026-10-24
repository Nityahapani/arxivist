from swin_repro.train import run
from swin_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["top1"] <= 1
