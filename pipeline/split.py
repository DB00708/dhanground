"""Assign every corpus document to train, dev or test. Usage: python -m pipeline.split corpus/ data/splits.json"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from core.splits import split_docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/splits.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    doc_ids = [p.stem for p in sorted(a.corpus_dir.glob("*.jsonl"))]
    splits = split_docs(doc_ids, cfg["doc_ratios"], cfg["seed"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
