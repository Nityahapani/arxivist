from detr_repro.train import run
from detr_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert s["final_loss"] > 0 and 0 <= s["recall_iou50"] <= 1
