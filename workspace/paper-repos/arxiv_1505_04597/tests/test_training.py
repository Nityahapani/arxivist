from unet_repro.train import run
from unet_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["eval_iou"] <= 1 and s["output_size"] == 52
