"""Thin wrapper over CuratorKIT: minhash dedup, grounded-hallucination gate and reward gate on our rows.

Rows go in as Alpaca records (instruction = question, input = gold passage text, output = answer):
CuratorKIT's hallucination gate reads the source text from `input`. Passed rows are matched back by
a hash of (instruction, output), so cleaning is off to keep text byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def row_key(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\x1f{answer}".encode("utf-8")).hexdigest()


def run_gates(rows: list[dict], work_dir: Path, judge_model: str | None,
              hallucination_threshold: float | None, reward_threshold: float | None) -> tuple[dict, dict]:
    """Returns ({row_key: reward_score_or_None} for passed rows, {row_key: rejection reason})."""
    from curatorkit import Curator, CuratorConfig

    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / "input.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for r in rows:
            gold = [p["text"] for p in r["passages"] if p["pid"] in (r["cited_pids"] or [r.get("source_pid")])]
            f.write(json.dumps({
                "instruction": r["question"],
                "input": "\n".join(gold),
                "output": r["answer"],
            }, ensure_ascii=False) + "\n")

    use_judge = judge_model is not None and (hallucination_threshold is not None or reward_threshold is not None)
    result = Curator(CuratorConfig(
        dataset=str(src),
        dedup="minhash",
        clean=False,
        min_tokens=3,
        max_tokens=4096,
        llm_model=judge_model if use_judge else None,
        judge_llm_model=judge_model if use_judge else None,
        hallucination_threshold=hallucination_threshold if use_judge else None,
        reward_threshold=reward_threshold if use_judge else None,
        export_formats=["alpaca"],
        output_dir=str(work_dir / "out"),
        name="dhanground-curation",
    )).run()

    passed = {}
    for s in result.passed:
        score = s.label if isinstance(s.label, (int, float)) else None
        passed[row_key(s.instruction, s.output)] = score
    rejected = {row_key(s.instruction, s.output): getattr(s, "rejection_reason", "rejected") for s in result.rejected}
    return passed, rejected
