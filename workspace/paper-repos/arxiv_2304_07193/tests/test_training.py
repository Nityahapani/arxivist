from dinov2_repro.train import run
from dinov2_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["knn_top1"] <= 1 and s["final_loss"] > 0
