"""Shared bits for training adapters: configs, adapter naming, checkpoint sync to the HF Hub."""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from pathlib import Path

import yaml


def load_configs() -> tuple[dict, dict]:
    arms = yaml.safe_load(Path("configs/arms.yaml").read_text(encoding="utf-8"))
    lora = yaml.safe_load(Path("configs/lora.yaml").read_text(encoding="utf-8"))
    return arms, lora


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def adapter_name(arm: str, seed: int) -> str:
    return f"{arm}-s{seed}-{git_sha()}"


class CheckpointSync:
    """Uploads each new checkpoint-* folder under output_dir to the Hub, so a dead GPU session can resume.

    On start it reads which checkpoints the repo already holds complete and never uploads those again
    (a resumed run restores one of them into output_dir).
    """

    def __init__(self, output_dir: Path, repo_id: str | None, every_s: int = 60):
        self.output_dir, self.repo_id, self.every_s = Path(output_dir), repo_id, every_s
        self.sent: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        if self.repo_id:
            from huggingface_hub import create_repo, list_repo_files

            create_repo(self.repo_id, private=True, exist_ok=True)
            self.sent |= {f"checkpoint-{n}" for n in complete_checkpoint_steps(list_repo_files(self.repo_id))}
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.repo_id:
            self._stop.set()
            self._thread.join()
            self.sync()

    def _loop(self):
        while not self._stop.wait(self.every_s):
            self.sync()

    def sync(self):
        from huggingface_hub import upload_folder

        for ckpt in sorted(self.output_dir.glob("checkpoint-*")):
            # transformers writes trainer_state.json last, so its presence marks a finished save
            if ckpt.name in self.sent or not (ckpt / "trainer_state.json").exists():
                continue
            upload_folder(repo_id=self.repo_id, folder_path=str(ckpt), path_in_repo=f"checkpoints/{ckpt.name}")
            self.sent.add(ckpt.name)


_CKPT_STATE = re.compile(r"^checkpoints/checkpoint-(\d+)/trainer_state\.json$")


def complete_checkpoint_steps(repo_files: list[str]) -> set[int]:
    """Every N with checkpoints/checkpoint-N/trainer_state.json present (a half-uploaded folder lacks it)."""
    return {int(m.group(1)) for f in repo_files if (m := _CKPT_STATE.match(f))}


def latest_checkpoint_step(repo_files: list[str]) -> int | None:
    steps = complete_checkpoint_steps(repo_files)
    return max(steps) if steps else None


def resume_checkpoint(output_dir: Path) -> Path | None:
    """Newest local checkpoint-N under output_dir that finished saving (has trainer_state.json).

    Passed to trainer.train() as an explicit path: resume_from_checkpoint=True would make transformers pick
    the highest-numbered folder even when a crash left it half-written.
    """
    best: tuple[int, Path] | None = None
    for d in Path(output_dir).glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and (d / "trainer_state.json").is_file() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), d)
    return best[1] if best else None


def name_from_repo(repo_id: str, arm: str, seed: int) -> str:
    """Adapter name of the run that owns repo_id; refuses a repo made for another arm or seed."""
    name = repo_id.split("/")[-1].removeprefix("dhanground-")
    if not name.startswith(f"{arm}-s{seed}-"):
        raise SystemExit(f"{repo_id} is not a {arm} seed {seed} run (expected <org>/dhanground-{arm}-s{seed}-<sha>)")
    return name


def restore_latest_checkpoint(repo_id: str, output_dir: Path) -> Path | None:
    """Download the newest complete Hub checkpoint into output_dir/checkpoint-N; None if the repo has none."""
    from huggingface_hub import list_repo_files, snapshot_download

    step = latest_checkpoint_step(list_repo_files(repo_id))
    if step is None:
        return None
    output_dir = Path(output_dir)
    dst = output_dir / f"checkpoint-{step}"
    if (dst / "trainer_state.json").exists():
        return dst
    staging = output_dir / ".hub_restore"
    snapshot_download(repo_id=repo_id, local_dir=str(staging), allow_patterns=[f"checkpoints/checkpoint-{step}/*"])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(staging / "checkpoints" / dst.name), str(dst))
    shutil.rmtree(staging, ignore_errors=True)
    return dst


def push_adapter(folder: Path, repo_id: str) -> None:
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, private=True, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=str(folder), ignore_patterns=["checkpoint-*", "*.pt", "*.bin.index"])
