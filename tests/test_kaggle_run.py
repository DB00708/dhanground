import json
import py_compile

from scripts.kaggle_run import BASE_MODEL_SOURCE, TASKS, build_kernel, stage_dir
from scripts.smoke_train import verdict


def test_verdict_go_when_loss_falls_and_finite():
    v = verdict([2.0, 1.9, 1.8, 1.6, 1.5, 1.4, 1.2, 1.1])
    assert v["go"] and v["finite"] and v["loss_last"] < v["loss_first"]


def test_verdict_no_go_on_nan_or_flat():
    assert not verdict([2.0, float("nan"), 1.0])["go"]
    assert not verdict([1.0, 1.0, 1.0, 1.0])["go"]
    assert not verdict([])["go"]


def test_kernel_needs_no_hub_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETSHOULDNOTAPPEAR")
    for task in TASKS:
        d = tmp_path / task
        d.mkdir()
        build_kernel(task, "a" * 40, "kuser", d)
        meta = json.loads((d / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
        assert meta["id"] == "kuser/dhanground-gpu" and meta["machine_shape"] == "NvidiaTeslaT4"
        assert meta["model_sources"] == [BASE_MODEL_SOURCE] and meta["dataset_sources"] == ["kuser/dhanground-work"]
        script = (d / "run.py").read_text()
        py_compile.compile(str(d / "run.py"), doraise=True)
        assert "a" * 40 in script and "DHANGROUND_PRECISION" in script and "DHANGROUND_BASE_PATH" in script
        assert "hf_SECRET" not in script and "HF_TOKEN" not in script and "--hub-org" not in script


def test_seed0_trains_both_arms_in_parallel_with_aligntune():
    steps = TASKS["sft-seed0"]["steps"]
    assert TASKS["sft-seed0"]["parallel"] and len(steps) == 2
    assert {s[s.index("--arm") + 1] for s in steps} == {"A1", "A2"}
    assert all(s[s.index("--backend") + 1] == "aligntune" and s[s.index("--seed") + 1] == "0" for s in steps)


def test_stage_dir_is_private_and_trims_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for src in ("data/raw/train.jsonl", "data/arms/A1/train.jsonl", "data/arms/A2/train.jsonl"):
        p = tmp_path / src
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join('{"i": %d}\n' % i for i in range(1000)))
    out = stage_dir(tmp_path / "stage", "kuser")
    assert len((out / "smoke/train.jsonl").read_text().splitlines()) == 400
    assert len((out / "arms/A2/train.jsonl").read_text().splitlines()) == 1000
    assert json.loads((out / "dataset-metadata.json").read_text())["id"] == "kuser/dhanground-work"
