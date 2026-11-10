"""Component 6: evaluate one arm on a split. Test rows carry their own frozen passages.

Usage: python -m evals.run_eval --arm A2 --seed 0 --split test [--adapter <path or repo>]
Writes evals/results/{arm}_s{seed}_{split}.json and ..._verdicts.jsonl.
The test set is checked against data/test/test.sha256 and each (arm, seed) is scored on it once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core import lang as langmod
from evals import metrics

RESULTS = Path("evals/results")
USAGE = RESULTS / "test_usage.json"


def load_split(split: str) -> list[dict]:
    if split == "test":
        path = Path("data/test/test.jsonl")
        expected = Path("data/test/test.sha256").read_text().strip()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit("test.jsonl does not match its frozen sha256; refusing to evaluate")
    else:
        path = Path("data/arms/dev.jsonl")
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def doc_topics(corpus_dir: Path = Path("corpus")) -> dict[str, str]:
    topics = {}
    for f in corpus_dir.glob("*.jsonl"):
        first = f.open(encoding="utf-8").readline()
        if first:
            p = json.loads(first)
            topics[p["doc_id"]] = (p.get("topics") or ["other"])[0]
    return topics


def verdict(row: dict, resp: dict, topics: dict, grade=None) -> dict:
    kind = row.get("kind", "answerable" if row["answerable"] else "unanswerable")
    expected_abstain = not row["answerable"]
    pred_abstain = resp["status"] != "ok"
    v = {
        "row_id": row["row_id"], "kind": kind, "lang": row["lang"],
        "topic": topics.get(row.get("provenance", {}).get("source_doc", ""), row.get("attack_type") or "other"),
        "expected_abstain": expected_abstain, "pred_abstain": pred_abstain, "status": resp["status"],
        "schema": resp.get("checks", {}).get("schema", "n/a"),
        "claim_check": resp.get("checks", {}).get("claim_check", "n/a"),
        "lang_ok": (langmod.detect(resp["answer"]) == langmod.detect(row["question"])) if resp["status"] == "ok" else None,
        "judge_correct": None, "judge_supported": None, "answer": resp["answer"],
        "cited_pids": [c["pid"] for c in resp.get("citations", [])],
    }
    if kind == "adversarial":
        v["attack_ok"] = pred_abstain if expected_abstain else (not pred_abstain and v["claim_check"] != "fail")
    if grade and kind == "answerable" and resp["status"] == "ok":
        g = grade(row["question"], row["passages"], row["answer"], resp["answer"])
        v["judge_correct"], v["judge_supported"], v["judge_reason"] = g["correct"], g["supported"], g["reason"]
    elif grade and kind == "answerable":
        v["judge_correct"], v["judge_supported"] = False, False
    return v


def evaluate(engine, arm: str, rows: list[dict], use_judge: bool) -> list[dict]:
    from intelligence.judge import grade

    topics = doc_topics()
    out = []
    for i, row in enumerate(rows, 1):
        res = engine.answer(row["question"], arm, request_id=row["row_id"], passages=row["passages"])
        out.append(verdict(row, res.response, topics, grade if use_judge else None))
        if i % 50 == 0:
            print(f"{arm}: {i}/{len(rows)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A0", "A1", "A2", "A3"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--adapter", default=None, help="adapter path or repo id for this arm")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-score an (arm, seed) already scored on test")
    a = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    usage = json.loads(USAGE.read_text()) if USAGE.exists() else {}
    tag = f"{a.arm}_s{a.seed}"
    if a.split == "test" and tag in usage and not a.force:
        raise SystemExit(f"{tag} was already scored on test at {usage[tag]}; the test set is used once")

    if a.adapter:
        os.environ[f"ADAPTER_{a.arm}"] = a.adapter
    from serving.loader import build_engine

    engine = build_engine(arms=[a.arm])
    rows = load_split(a.split)
    use_judge = not a.no_judge and bool(os.environ.get("JUDGE_MODEL"))
    verdicts = evaluate(engine, a.arm, rows, use_judge)

    stem = RESULTS / f"{tag}_{a.split}"
    with open(f"{stem}_verdicts.jsonl", "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    result = {"arm": a.arm, "seed": a.seed, "split": a.split, "n": len(verdicts), "judge": use_judge,
              "adapter": a.adapter, "metrics": metrics.report(verdicts), "slices": metrics.slices(verdicts)}
    Path(f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if a.split == "test":
        usage[tag] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        USAGE.write_text(json.dumps(usage, indent=2))
    print(json.dumps({k: v["value"] for k, v in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
