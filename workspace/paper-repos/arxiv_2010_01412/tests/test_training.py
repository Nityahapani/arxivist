from sam_repro.train import run
from sam_repro.utils import load_config


def test_smoke_sam_and_sgd(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"]["steps"] = 2
    for opt in ("sam", "sgd"):
        cfg["train"]["optimizer"] = opt
        cfg["run_name"] = opt
        s = run(cfg, tmp_path, smoke=True)
        assert 0 <= s["test_acc"] <= 1
    cfg["train"].update(optimizer="sam", m_sharpness=8)
    assert run(cfg, tmp_path, smoke=True)["steps"] == 2
