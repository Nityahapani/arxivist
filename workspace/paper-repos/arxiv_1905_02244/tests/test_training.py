from mobilenetv3_repro.train import run
from mobilenetv3_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    assert 0 <= run(cfg, tmp_path, smoke=True)["top1"] <= 1
