"""Assign every corpus document to train, dev or test, balanced by topic and passage count.

Usage: python -m pipeline.split corpus/ data/splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from core.splits import split_docs_stratified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/splits.yaml"))
    ap.add_argument("--sources", type=Path, default=Path("configs/rbi_sources.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    sources = yaml.safe_load(a.sources.read_text(encoding="utf-8"))["documents"]
    sizes = {p.stem: sum(1 for _ in p.open(encoding="utf-8")) for p in sorted(a.corpus_dir.glob("*.jsonl"))}
    topics = {d["doc_id"]: d.get("topics", []) for d in sources if d["doc_id"] in sizes}
    splits = split_docs_stratified(topics, sizes, cfg["core_topics"], cfg["passage_ratios"], cfg["seed"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    for name in ("train", "dev", "test"):
        docs = splits[name]
        covered = sorted({t for d in docs for t in topics[d] if t in cfg["core_topics"]})
        print(f"{name}: {len(docs)} docs, {sum(sizes[d] for d in docs)} passages, core topics {covered}")


if __name__ == "__main__":
    main()
