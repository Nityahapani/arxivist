from qlora_repro.train import run
from qlora_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert s["trainable_params"] < s["base_params"] and s["quant_mse"]["nf4"] > 0
