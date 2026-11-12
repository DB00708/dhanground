"""Component 3: curation gate. Raw rows -> curated rows + gate log.

Deterministic gates run first (cheap, re-runnable by anyone), then CuratorKIT's dedup and LLM gates.
Usage: python -m pipeline.curate --raw data/raw --corpus corpus --out data/curated
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pydantic import ValidationError

from adapters import curatorkit_run
from core import claimcheck, lang as langmod
from core.schema import MAX_WORDS, TrainingRow
from intelligence import judge

HALLUCINATION_THRESHOLD = 0.7
REWARD_THRESHOLD = 0.7


def deterministic_gates(row: dict, corpus_sha: dict[str, str]) -> list[tuple[str, str, str]]:
    """Returns [(gate, decision, reason)]; decision is pass or fail."""
    out = []
    try:
        TrainingRow.model_validate(row)
        out.append(("schema", "pass", ""))
    except ValidationError as e:
        return [("schema", "fail", str(e.errors()[0]["msg"]))]

    srcs = row.get("source_pids") or [row.get("source_pid")]
    shas = row["provenance"].get("span_sha256s") or [row["provenance"]["span_sha256"]]
    prov = row["provenance"]
    prov_ok = (len(srcs) == len(shas) and all(corpus_sha.get(s) == h for s, h in zip(srcs, shas))
               and srcs[0] == row.get("source_pid") and shas[0] == prov["span_sha256"])   # legacy fields = first source
    out.append(("provenance", "pass" if prov_ok else "fail", "" if prov_ok else "span hash mismatch"))

    q_lang = langmod.detect(row["question"])
    out.append(("lang_question", "pass" if q_lang == row["lang"] else "fail", q_lang))

    if row["answerable"]:
        words = len(row["answer"].split())
        cited_ok = bool(row["cited_pids"]) and all(s in row["cited_pids"] for s in srcs)
        out.append(("citations", "pass" if cited_ok else "fail", "" if cited_ok else "source passage not cited"))
        out.append(("length", "pass" if 0 < words <= MAX_WORDS else "fail", str(words)))
        a_lang = langmod.detect(row["answer"])
        out.append(("lang_answer", "pass" if a_lang == row["lang"] else "fail", a_lang))
        cited = [p["text"] for p in row["passages"] if p["pid"] in row["cited_pids"]]
        cc = claimcheck.check(row["answer"], cited)
        out.append(("claimcheck", "fail" if cc.status == claimcheck.FAIL else "pass", ",".join(cc.unsupported)))
    return out


def curate(raw_dir: Path, corpus_dir: Path, out_dir: Path, use_llm_gates: bool = True) -> None:
    corpus_sha = {}
    for f in corpus_dir.glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            p = json.loads(line)
            corpus_sha[p["pid"]] = p["sha256"]

    judge_model = os.environ.get("JUDGE_MODEL") if use_llm_gates else None
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "gate_log.jsonl").open("w", encoding="utf-8")

    def record(row, gate, decision, reason="", score=None):
        log.write(json.dumps({"row_id": row["row_id"], "split": row["split"], "gate": gate,
                              "decision": decision, "reason": reason, "score": score},
                             ensure_ascii=False) + "\n")

    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        split = raw_file.stem
        rows = [json.loads(l) for l in raw_file.open(encoding="utf-8")]
        survivors = []
        for row in rows:
            results = deterministic_gates(row, corpus_sha)
            for gate, decision, reason in results:
                record(row, gate, decision, reason)
            if all(d == "pass" for _, d, _ in results):
                survivors.append(row)

        if judge_model:
            kept = []
            for row in survivors:
                if row["answerable"]:
                    kept.append(row)
                    continue
                try:
                    answered = judge.passages_answer(row["question"], row["passages"])
                except Exception as e:
                    record(row, "unanswerable_verified", "fail", f"judge_error:{type(e).__name__}")
                    continue
                record(row, "unanswerable_verified", "fail" if answered else "pass",
                       "passages answer it" if answered else "")
                if not answered:
                    kept.append(row)
            survivors = kept

        by_key = {curatorkit_run.row_key(r["question"], r["answer"]): r for r in survivors}
        answerable = [r for r in survivors if r["answerable"]]
        unanswerable = [r for r in survivors if not r["answerable"]]
        passed: dict[str, float | None] = {}
        rejected: dict[str, str] = {}
        for group, rows_g, thresholds in [
            ("answerable", answerable, (HALLUCINATION_THRESHOLD, REWARD_THRESHOLD)),
            ("unanswerable", unanswerable, (None, None)),
        ]:
            if not rows_g:
                continue
            p, rj = curatorkit_run.run_gates(rows_g, out_dir / "curatorkit" / f"{split}_{group}",
                                             judge_model, *thresholds)
            passed.update(p)
            rejected.update(rj)

        curated = []
        for key, row in by_key.items():
            if key in passed:
                record(row, "curatorkit", "pass", score=passed[key])
                row = {**row, "provenance": {**row["provenance"], "gates": {
                    "dedup": "pass", "quality": passed[key] if passed[key] is not None else 1.0,
                    "halluc": "pass" if row["answerable"] and judge_model else "n/a", "provenance": "pass"}}}
                curated.append(row)
            else:
                record(row, "curatorkit", "fail", rejected.get(key, "dropped"))
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in curated:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} raw -> {len(survivors)} after deterministic+verify -> {len(curated)} curated")
    log.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/curated"))
    ap.add_argument("--no-llm-gates", action="store_true", help="deterministic gates + dedup only")
    a = ap.parse_args()
    curate(a.raw, a.corpus, a.out, use_llm_gates=not a.no_llm_gates)


if __name__ == "__main__":
    main()
