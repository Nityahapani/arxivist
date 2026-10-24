from dpo_repro.train import run
from dpo_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 3
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["dpo_true_reward"] <= 1 and s["final_dpo_loss"] > 0
