import json

from lora_repro.train import run
from lora_repro.utils import load_config


def test_smoke_pipeline(tmp_path):
    cfg = load_config("configs/smoke.json")
    cfg["train"].update(pretrain_steps=5, steps=5, eval_every=5)
    s = run(cfg, tmp_path)
    assert s["trainable_params"] < s["total_params"]
    assert s["merge_max_abs_diff"] < 1e-4
    assert (tmp_path / "smoke" / "summary.json").exists()
    assert json.loads((tmp_path / "smoke" / "summary.json").read_text())["is_paper_result"] is False
