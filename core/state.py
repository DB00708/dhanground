"""Stage manifests: each pipeline stage writes state.json with its input hash and skips work when it matches."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STATES = ("INGESTED", "GENERATED", "CURATED", "SPLIT", "TRAINED", "MINED", "EVALUATED", "PUBLISHED")


def hash_inputs(paths: list[Path], params: dict | None = None) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and f.name != "state.json":
                h.update(str(f.relative_to(p.parent)).replace("\\", "/").encode())
                h.update(f.read_bytes())
    h.update(json.dumps(params or {}, sort_keys=True).encode())
    return h.hexdigest()


def is_fresh(out_dir: Path, input_hash: str) -> bool:
    manifest = Path(out_dir) / "state.json"
    if not manifest.exists():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("input_hash") == input_hash


def write_manifest(out_dir: Path, state: str, input_hash: str, **extra) -> None:
    if state not in STATES:
        raise ValueError(f"unknown state {state}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "state": state,
        "input_hash": input_hash,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    (out_dir / "state.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
