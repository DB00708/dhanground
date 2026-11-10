import sys
import types

from adapters.common import latest_checkpoint_step, restore_latest_checkpoint

REPO = "org/dhanground-A2-s0-abc1234"


def test_picks_highest_complete_step_numerically():
    files = [
        "adapter_config.json",
        "checkpoints/checkpoint-800/trainer_state.json",
        "checkpoints/checkpoint-800/adapter_model.safetensors",
        "checkpoints/checkpoint-1000/trainer_state.json",
        "checkpoints/checkpoint-200/trainer_state.json",
    ]
    assert latest_checkpoint_step(files) == 1000


def test_ignores_folders_without_trainer_state():
    files = [
        "checkpoints/checkpoint-400/trainer_state.json",
        "checkpoints/checkpoint-600/adapter_model.safetensors",  # upload died before trainer_state.json
    ]
    assert latest_checkpoint_step(files) == 400


def test_no_checkpoints_returns_none():
    assert latest_checkpoint_step(["README.md", "checkpoint-200/trainer_state.json"]) is None


def _fake_hub(monkeypatch, files, calls):
    def list_repo_files(repo_id):
        calls.append(("list", repo_id))
        return files

    def snapshot_download(repo_id, local_dir, allow_patterns):
        calls.append(("download", repo_id, allow_patterns))
        from pathlib import Path

        for f in files:
            if any(f.startswith(p.rstrip("*")) for p in allow_patterns):
                dst = Path(local_dir) / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text("{}")
        return local_dir

    mod = types.SimpleNamespace(list_repo_files=list_repo_files, snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_restore_downloads_only_latest_into_output_dir(monkeypatch, tmp_path):
    files = [
        "checkpoints/checkpoint-800/trainer_state.json",
        "checkpoints/checkpoint-1000/trainer_state.json",
        "checkpoints/checkpoint-1000/adapter_model.safetensors",
        "checkpoints/checkpoint-1200/adapter_model.safetensors",
    ]
    calls = []
    _fake_hub(monkeypatch, files, calls)
    out = tmp_path / "checkpoints" / "A2-s0-other"

    got = restore_latest_checkpoint(REPO, out)

    assert got == out / "checkpoint-1000"
    assert (got / "trainer_state.json").exists() and (got / "adapter_model.safetensors").exists()
    assert ("download", REPO, ["checkpoints/checkpoint-1000/*"]) in calls
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint-1000"]  # staging dir cleaned up


def test_restore_returns_none_when_repo_has_no_checkpoint(monkeypatch, tmp_path):
    calls = []
    _fake_hub(monkeypatch, ["adapter_config.json"], calls)
    assert restore_latest_checkpoint(REPO, tmp_path / "out") is None
    assert all(c[0] != "download" for c in calls)


def test_restore_skips_download_when_already_local(monkeypatch, tmp_path):
    calls = []
    _fake_hub(monkeypatch, ["checkpoints/checkpoint-400/trainer_state.json"], calls)
    local = tmp_path / "checkpoint-400"
    local.mkdir()
    (local / "trainer_state.json").write_text("{}")
    assert restore_latest_checkpoint(REPO, tmp_path) == local
    assert all(c[0] != "download" for c in calls)
