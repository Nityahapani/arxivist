from flashattention_repro.train import run
from flashattention_repro.utils import load_config


def test_smoke_flash_matches_standard(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert s["loss_abs_diff"] < 1e-4 and s["hbm_access_ratio_standard_over_flash"] > 1
