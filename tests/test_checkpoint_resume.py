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


# --- local resume picker, Hub-aware sync, run naming (edge case 19) ---

import pytest

from adapters.common import CheckpointSync, name_from_repo, resume_checkpoint


def _ckpt(root, step, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if complete:
        (d / "trainer_state.json").write_text("{}")
    return d


def test_resume_checkpoint_skips_higher_numbered_partial(tmp_path):
    _ckpt(tmp_path, 200)
    want = _ckpt(tmp_path, 1000)
    _ckpt(tmp_path, 1200, complete=False)  # crash mid-save; transformers' get_last_checkpoint would pick this
    assert resume_checkpoint(tmp_path) == want


def test_resume_checkpoint_none_when_nothing_complete(tmp_path):
    _ckpt(tmp_path, 400, complete=False)
    assert resume_checkpoint(tmp_path) is None
    assert resume_checkpoint(tmp_path / "missing") is None


def test_sync_never_reuploads_checkpoints_already_on_hub(monkeypatch, tmp_path):
    uploads = []
    hub_files = ["checkpoints/checkpoint-800/trainer_state.json", "checkpoints/checkpoint-1000/trainer_state.json"]
    mod = types.SimpleNamespace(
        create_repo=lambda *a, **k: None,
        list_repo_files=lambda repo_id: hub_files,
        upload_folder=lambda repo_id, folder_path, path_in_repo: uploads.append(path_in_repo),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    _ckpt(tmp_path, 1000)  # restored from the Hub
    sync = CheckpointSync(tmp_path, REPO, every_s=3600)
    with sync:
        _ckpt(tmp_path, 1200)  # written after resume
        _ckpt(tmp_path, 1400, complete=False)  # still being saved
    assert uploads == ["checkpoints/checkpoint-1200"]


def test_name_from_repo_keeps_original_sha_and_checks_arm_seed():
    assert name_from_repo("org/dhanground-A2-s1-abc1234", "A2", 1) == "A2-s1-abc1234"
    with pytest.raises(SystemExit):
        name_from_repo("org/dhanground-A2-s0-abc1234", "A2", 1)
    with pytest.raises(SystemExit):
        name_from_repo("org/dhanground-A1-s0-abc1234", "A2", 0)


def test_sync_survives_failed_upload_and_retries(monkeypatch, tmp_path):
    uploads, fail = [], {"left": 1}

    def upload_folder(repo_id, folder_path, path_in_repo):
        if fail["left"]:
            fail["left"] -= 1
            raise OSError("hub down")
        uploads.append(path_in_repo)

    mod = types.SimpleNamespace(create_repo=lambda *a, **k: None, list_repo_files=lambda r: [],
                                upload_folder=upload_folder)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    _ckpt(tmp_path, 200)
    sync = CheckpointSync(tmp_path, REPO, every_s=3600)
    sync.sync()  # fails, must not raise
    assert uploads == [] and "checkpoint-200" not in sync.sent
    sync.sync()
    assert uploads == ["checkpoints/checkpoint-200"]


def test_exit_does_not_raise_when_final_upload_fails(monkeypatch, tmp_path):
    def upload_folder(**kw):
        raise OSError("hub down")

    mod = types.SimpleNamespace(create_repo=lambda *a, **k: None, list_repo_files=lambda r: [],
                                upload_folder=upload_folder)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    with CheckpointSync(tmp_path, REPO, every_s=3600):
        _ckpt(tmp_path, 200)
    # reaching here means the training script goes on to save and push the final adapter


def test_restore_replaces_partial_dst_and_stale_staging(monkeypatch, tmp_path):
    files = ["checkpoints/checkpoint-600/trainer_state.json", "checkpoints/checkpoint-600/adapter_model.safetensors"]
    calls = []
    _fake_hub(monkeypatch, files, calls)
    partial = tmp_path / "checkpoint-600"
    partial.mkdir()
    (partial / "optimizer.pt").write_text("half-written")  # no trainer_state.json
    stale = tmp_path / ".hub_restore" / "checkpoints" / "checkpoint-400"
    stale.mkdir(parents=True)
    (stale / "trainer_state.json").write_text("{}")  # left by an earlier crashed restore

    got = restore_latest_checkpoint(REPO, tmp_path)

    assert got == partial
    assert sorted(p.name for p in got.iterdir()) == ["adapter_model.safetensors", "trainer_state.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-600"]
