"""Component 4: DPO pairs mined from the curated-SFT model's own failures (ED8).

chosen = the gold answer JSON, rejected = what A2 actually produced. Failure kinds:
over_refusal, answered_unanswerable, claim_check, schema, judge (wrong or unsupported).

Usage: python -m intelligence.mine_pairs --a2 <adapter> [--extra-curated-train 1500]
Rows: dev split, plus optionally curated train rows that were not sampled into A2.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from core.template import Passage, render_assistant, render_user


def classify_failure(row: dict, resp: dict, grade=None) -> str | None:
    if resp["status"] == "timeout":
        return None
    if not row["answerable"]:
        return "answered_unanswerable" if resp["status"] == "ok" else None
    if resp["status"] != "ok":
        return "schema" if resp.get("abstain_reason") == "schema" else "over_refusal"
    if resp["checks"].get("claim_check") == "fail":
        return "claim_check"
    if grade:
        g = grade(row["question"], row["passages"], row["answer"], resp["answer"])
        if not (g["correct"] and g["supported"]):
            return "judge"
    return None


def to_pair(row: dict, rejected_raw: str, reason: str) -> dict:
    passages = [Passage(p["pid"], p["text"]) for p in row["passages"]]
    chosen = render_assistant(row["answerable"], row["answer"], row["cited_pids"], row["lang"])
    return {
        "row_id": row["row_id"],
        "prompt": [{"role": "user", "content": render_user(row["question"], passages)}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected_raw}],
        "reject_reason": reason,
    }


def candidate_rows(extra: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in Path("data/arms/dev.jsonl").open(encoding="utf-8")]
    if extra:
        used = {json.loads(l)["row_id"] for l in Path("data/arms/A2/rows.jsonl").open(encoding="utf-8")}
        leftover = [json.loads(l) for l in Path("data/curated/train.jsonl").open(encoding="utf-8")]
        leftover = sorted((r for r in leftover if r["row_id"] not in used), key=lambda r: r["row_id"])
        rows += random.Random(seed).sample(leftover, min(extra, len(leftover)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", required=True)
    ap.add_argument("--extra-curated-train", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/arms/A3_pairs.jsonl"))
    a = ap.parse_args()

    os.environ["ADAPTER_A2"] = a.a2
    from intelligence.judge import grade
    from serving.loader import build_engine

    engine = build_engine(arms=["A2"])
    use_judge = bool(os.environ.get("JUDGE_MODEL"))
    pairs, counts = [], {}
    rows = candidate_rows(a.extra_curated_train, seed=0)
    for i, row in enumerate(rows, 1):
        res = engine.answer(row["question"], "A2", passages=row["passages"])
        reason = classify_failure(row, res.response, grade if use_judge else None)
        if reason and res.raw_outputs:
            pairs.append(to_pair(row, res.raw_outputs[-1], reason))
            counts[reason] = counts.get(reason, 0) + 1
        if i % 50 == 0:
            print(f"{i}/{len(rows)} rows, {len(pairs)} pairs")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs from {len(rows)} rows: {counts}")


if __name__ == "__main__":
    main()
