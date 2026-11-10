"""Shared bits for training adapters: configs, adapter naming, checkpoint sync to the HF Hub."""
from __future__ import annotations

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
    """Uploads each new checkpoint-* folder under output_dir to the Hub, so a dead GPU session can resume."""

    def __init__(self, output_dir: Path, repo_id: str | None, every_s: int = 60):
        self.output_dir, self.repo_id, self.every_s = Path(output_dir), repo_id, every_s
        self.sent: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        if self.repo_id:
            from huggingface_hub import create_repo

            create_repo(self.repo_id, private=True, exist_ok=True)
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
            if ckpt.name in self.sent or not (ckpt / "trainer_state.json").exists():
                continue
            upload_folder(repo_id=self.repo_id, folder_path=str(ckpt), path_in_repo=f"checkpoints/{ckpt.name}")
            self.sent.add(ckpt.name)


def push_adapter(folder: Path, repo_id: str) -> None:
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, private=True, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=str(folder), ignore_patterns=["checkpoint-*", "*.pt", "*.bin.index"])
