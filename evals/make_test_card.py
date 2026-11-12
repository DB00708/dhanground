"""Write evals/test_set_card.md: how the frozen test rows were built and checked.

Usage: python -m evals.make_test_card [--test data/test/test.jsonl] [--review data/test/review.csv]
                                      [--corpus corpus] [--out evals/test_set_card.md]
Topic uses the same rule as evals.run_eval: the first topic of the row's source document,
else the attack type, else "other".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from evals.run_eval import doc_topics

PLACEHOLDER_MARK = "<!-- placeholder: run python -m evals.make_test_card -->"


def table(title: str, counts: Counter, total: int) -> list[str]:
    lines = [f"| {title} | Rows | Share |", "|---|---:|---:|"]
    for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        lines.append(f"| {key or '(none)'} | {n} | {n / total:.0%} |" if total else f"| {key} | {n} | - |")
    return lines + [""]


def review_counts(review: Path) -> Counter:
    if not review.exists():
        return Counter()
    with review.open(encoding="utf-8-sig", newline="") as f:
        return Counter((r.get("keep") or "").strip().lower() or "blank" for r in csv.DictReader(f))


def build_card(test: Path, review: Path, corpus: Path) -> str:
    rows = [json.loads(l) for l in test.open(encoding="utf-8") if l.strip()]
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    sha_file = test.with_name("test.sha256")
    frozen = sha_file.read_text(encoding="utf-8").strip() if sha_file.exists() else None
    topics = doc_topics(corpus)
    n = len(rows)
    kinds = Counter(r.get("kind", "answerable") for r in rows)
    langs = Counter(r.get("lang", "unknown") for r in rows)
    topic = Counter(topics.get(r.get("provenance", {}).get("source_doc", ""), r.get("attack_type") or "other")
                    for r in rows)
    attacks = Counter(r.get("attack_type") or "(none)" for r in rows if r.get("kind") == "adversarial")
    keep = review_counts(review)
    reviewed = keep.get("y", 0) + keep.get("n", 0)

    if frozen is None:
        hash_line = f"The file's sha256 is `{digest}`. No test.sha256 file was found next to it."
    elif frozen == digest:
        hash_line = f"The sha256 is `{digest}`. It matches test.sha256, so the file has not changed since the freeze."
    else:
        hash_line = (f"WARNING: the file's sha256 is `{digest}`, but test.sha256 says `{frozen}`. "
                     "The file changed after the freeze. Do not report results on it.")

    if keep:
        review_text = [
            f"A person read {reviewed} of {sum(keep.values())} candidate rows in `data/test/review.csv`.",
            f"They kept {keep.get('y', 0)} rows (keep=y) and removed {keep.get('n', 0)} rows (keep=n).",
            f"{sum(keep.values()) - reviewed} rows have no valid keep value.",
            "Only keep=y rows can go into the test set.",
        ]
    else:
        review_text = ["No review file was found. The rows were not hand-checked."]

    out = [
        "# DhanGround test set card",
        "",
        f"The frozen test set has {n} rows. This card says how they were built and checked.",
        "Run `python -m evals.make_test_card` again if the test set changes.",
        "",
        "## How the rows were made",
        "",
        "1. Generate. A teacher model wrote questions from RBI passages. The passages come only from",
        "   documents in the test split. The split is by document, so no test document is in train or dev.",
        "2. Curate. Each row passed our gates and the CuratorKIT gates (dedup, grounding, reward).",
        "   `gate_log.jsonl` in the dataset repo records each decision.",
        "3. Hand-check. A person read each candidate row with its source passage and marked keep=y or keep=n.",
        "4. Sample. With seed 0, the freeze step took up to 300 answerable and 100 unanswerable kept rows.",
        "5. Add attacks. People wrote the adversarial rows by hand in `evals/adversarial_test.jsonl`.",
        "   The freeze step attached the top 4 retrieved passages to each one.",
        "6. Freeze. The freeze step wrote `data/test/test.jsonl` and its sha256. Evaluation refuses to run",
        "   if the hash does not match.",
        "",
        "## Hand check",
        "",
        *review_text,
        "",
        "## Freeze",
        "",
        hash_line,
        "",
        "## Counts",
        "",
        *table("Kind", kinds, n),
        *table("Language", langs, n),
        "Topic is the first topic of the source document. Adversarial rows use their attack type.",
        "",
        *table("Topic", topic, n),
        *table("Attack type (adversarial rows)", attacks, sum(attacks.values())),
    ]
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("data/test/test.jsonl"))
    ap.add_argument("--review", type=Path, default=Path("data/test/review.csv"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--out", type=Path, default=Path("evals/test_set_card.md"))
    a = ap.parse_args()
    if not a.test.exists():
        raise SystemExit(f"{a.test} not found; run `python -m pipeline.build_arms freeze` first")
    a.out.write_text(build_card(a.test, a.review, a.corpus), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
