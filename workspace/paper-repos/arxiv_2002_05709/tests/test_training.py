from simclr_repro.train import run
from simclr_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    cfg["linear_eval"]["steps"] = 5
    s = run(cfg, tmp_path, smoke=True)
    assert 0.0 <= s["linear_eval_top1"] <= 1.0
    assert s["final_contrastive_loss"] > 0
