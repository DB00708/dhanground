"""Run intelligence.generate with the FakeTeacher and small row targets (e2e only; no API key).

Usage (cwd = e2e work dir, PYTHONPATH = repo): python -m tests.e2e.fake_generate --index index/e2e
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from intelligence.generate import generate
from tests.e2e.fakes import FakeTeacher

SMALL_TARGETS = {"train": 120, "dev": 24, "test": 40}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    cfg.update(targets=SMALL_TARGETS, concurrency=4)
    raise SystemExit(generate(a.index, a.splits, a.out, cfg, complete=FakeTeacher(), generator="fake-teacher@e2e"))


if __name__ == "__main__":
    main()
