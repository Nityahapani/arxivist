from segment_anything_repro.train import run
from segment_anything_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert s["final_loss"] > 0 and 0.0 <= s["final_mask_iou"] <= 1.0 and s["is_paper_result"] is False
