from alibi_repro.train import run
from alibi_repro.utils import load_config


def test_smoke(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 3
    s = run(cfg, tmp_path, smoke=True)
    assert set(s["runs"]) == {"alibi", "sinusoidal"}
    assert set(s["runs"]["alibi"]["valid_ppl_by_length"]) == {"32", "64", "128"}
