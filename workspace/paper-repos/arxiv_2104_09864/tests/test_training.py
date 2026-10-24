from roformer_repro.train import run
from roformer_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert set(s["results"]) == {"rope", "sinusoidal"}
