"""Pre-compute every demo example for all 4 arms, served when the GPU is down.

Usage: python -m apps.api.build_cache --index index/<version>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from apps.api.main import normalize
from serving.engine import ARMS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/demo_cache.json"))
    a = ap.parse_args()

    from serving.loader import build_engine

    engine = build_engine(index_dir=a.index)
    examples = yaml.safe_load(Path("configs/demo_examples.yaml").read_text(encoding="utf-8"))["examples"]
    cache = {}
    for ex in examples:
        cache[normalize(ex["question"])] = {
            arm: engine.answer(ex["question"], arm, request_id="cache", show_passages=True).response for arm in ARMS
        }
        print(ex["question"][:60], {arm: cache[normalize(ex["question"])][arm]["status"] for arm in ARMS})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
