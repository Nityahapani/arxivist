from ldm_repro.train import run
from ldm_repro.utils import load_config


def test_smoke_runs(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["ae_steps"] = 3
    cfg["train"]["steps"] = 2
    s = run(cfg, tmp_path, smoke=True)
    assert s["downsampling_f"] == 4 and s["sample_shape"] == [4, 3, 32, 32] and s["final_ldm_loss"] > 0
