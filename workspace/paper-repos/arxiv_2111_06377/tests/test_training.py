from mae_repro.train import run
from mae_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 3
    cfg["linear_probe"]["steps"] = 5
    s = run(cfg, tmp_path, smoke=True)
    assert s["final_recon_loss"] > 0 and 0 <= s["linear_probe_top1"] <= 1
