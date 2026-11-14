import json
import py_compile

from scripts.kaggle_run import TASKS, build_kernel
from scripts.smoke_train import verdict


def test_verdict_go_when_loss_falls_and_finite():
    v = verdict([2.0, 1.9, 1.8, 1.6, 1.5, 1.4, 1.2, 1.1])
    assert v["go"] and v["finite"] and v["loss_last"] < v["loss_first"]


def test_verdict_no_go_on_nan_or_flat():
    assert not verdict([2.0, float("nan"), 1.0])["go"]
    assert not verdict([1.0, 1.0, 1.0, 1.0])["go"]
    assert not verdict([])["go"]


def test_kernel_pins_commit_gpu_fp16_and_never_embeds_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETSHOULDNOTAPPEAR")
    for task in TASKS:
        d = tmp_path / task
        d.mkdir()
        build_kernel(task, "a" * 40, "someone", "kuser", d)
        meta = json.loads((d / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
        assert meta["id"] == "kuser/dhanground-gpu" and meta["machine_shape"] == "NvidiaTeslaT4"
        script = (d / "run.py").read_text()
        py_compile.compile(str(d / "run.py"), doraise=True)
        assert "a" * 40 in script and "DHANGROUND_PRECISION" in script and "UserSecretsClient" in script
        assert "hf_SECRET" not in script and "someone/dhanground-work" in script


def test_seed0_trains_both_arms_in_parallel_with_aligntune():
    steps = TASKS["sft-seed0"]["steps"]
    assert TASKS["sft-seed0"]["parallel"] and len(steps) == 2
    assert {s[s.index("--arm") + 1] for s in steps} == {"A1", "A2"}
    assert all(s[s.index("--backend") + 1] == "aligntune" and s[s.index("--seed") + 1] == "0" for s in steps)
