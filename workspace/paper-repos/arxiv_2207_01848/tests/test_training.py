from tabpfn_repro.train import run
from tabpfn_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert 0 <= s["tabpfn_auc"] <= 1 and s["final_prior_nll"] > 0
