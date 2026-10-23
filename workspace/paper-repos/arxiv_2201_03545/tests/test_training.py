from convnext_repro.train import run
from convnext_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 3
    cfg["train"]["ema_decay"] = 0.9
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["top1"] <= 1
