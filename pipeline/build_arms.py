"""Build arm datasets and the frozen test set.

  python -m pipeline.build_arms arms            # A1 (raw) and A2 (curated), same N, same strata
  python -m pipeline.build_arms review          # write data/test/review.csv for the hand check
  python -m pipeline.build_arms freeze --index index/<v>   # freeze test.jsonl + its sha256

A1 and A2 are sampled to identical counts per (language, answerable) stratum, so the only
difference between them is whether rows passed curation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from core.splits import check_no_overlap
from core.template import Passage, to_sft_messages

DATA = Path("data")
TEST_MIX = {"answerable": 300, "unanswerable": 100, "adversarial": 100}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratum(row: dict) -> tuple[str, bool]:
    return row["lang"], row["answerable"]


def stratified_targets(pool: list[dict], n: int) -> dict:
    counts = Counter(stratum(r) for r in pool)
    total = sum(counts.values())
    targets = {s: int(n * c / total) for s, c in counts.items()}
    for s, _ in counts.most_common(n - sum(targets.values())):
        targets[s] += 1
    return targets


def sample_to(pool: list[dict], targets: dict, rng: random.Random, label: str) -> list[dict]:
    by = {}
    for r in pool:
        by.setdefault(stratum(r), []).append(r)
    out, short = [], []
    for s, n in sorted(targets.items()):
        rows = sorted(by.get(s, []), key=lambda r: r["row_id"])
        if len(rows) < n:
            short.append(f"{s}: need {n}, have {len(rows)}")
            continue
        out.extend(rng.sample(rows, n))
    if short:
        raise SystemExit(f"{label} is short; generate more rows and re-curate: " + "; ".join(short))
    return out


def to_sft(row: dict) -> dict:
    passages = [Passage(p["pid"], p["text"]) for p in row["passages"]]
    return {"row_id": row["row_id"], "messages": to_sft_messages(row["question"], passages, row)}


def _leak_keys(row: dict) -> set[str]:
    from core.splits import normalize_question

    prov = row["provenance"]
    spans = {prov["span_sha256"], *(prov.get("span_sha256s") or [])}
    return {f"span:{s}" for s in spans} | {"q:" + hashlib.sha256(normalize_question(row["question"]).encode()).hexdigest()}


def drop_cross_split(rows: list[dict], protected: list[dict]) -> tuple[list[dict], int]:
    """Drop rows sharing a source paragraph or a normalised question with a higher-priority split.

    Splits are by document, but RBI repeats boilerplate across documents and the teacher can write the same
    question from two documents; test keeps its rows, dev yields to test, train yields to both.
    """
    seen = set().union(*(_leak_keys(r) for r in protected)) if protected else set()
    kept = [r for r in rows if not (_leak_keys(r) & seen)]
    return kept, len(rows) - len(kept)


def build_arms(n: int, seed: int) -> None:
    test = read_jsonl(DATA / "curated" / "test.jsonl")
    dev, d_dev = drop_cross_split(read_jsonl(DATA / "curated" / "dev.jsonl"), test)
    raw, d_raw = drop_cross_split(read_jsonl(DATA / "raw" / "train.jsonl"), test + dev)
    curated, d_cur = drop_cross_split(read_jsonl(DATA / "curated" / "train.jsonl"), test + dev)
    if d_dev or d_raw or d_cur:
        print(f"dropped cross-split duplicates: dev {d_dev}, raw train {d_raw}, curated train {d_cur}")
    targets = stratified_targets(raw, n)
    a1 = sample_to(raw, targets, random.Random(seed), "A1 raw pool")
    a2 = sample_to(curated, targets, random.Random(seed), "A2 curated pool")
    check_no_overlap(a1 + dev + test)
    check_no_overlap(a2 + dev + test)
    for arm, rows in (("A1", a1), ("A2", a2)):
        write_jsonl(DATA / "arms" / arm / "train.jsonl", [to_sft(r) for r in rows])
        write_jsonl(DATA / "arms" / arm / "rows.jsonl", rows)
    write_jsonl(DATA / "arms" / "dev.jsonl", dev)
    print(f"A1={len(a1)} A2={len(a2)} dev={len(dev)}; strata={dict(sorted(targets.items()))}")


def source_text(row: dict) -> str:
    """Text of every gold passage (two for multi-passage rows), so the reviewer sees all the evidence."""
    by_pid = {p["pid"]: p["text"] for p in row["passages"]}
    return "\n---\n".join(by_pid.get(pid, "") for pid in row.get("source_pids") or [row.get("source_pid")])


def write_review() -> None:
    rows = read_jsonl(DATA / "curated" / "test.jsonl")
    path = DATA / "test" / "review.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "lang", "kind", "question", "answer", "source_passage", "keep"])
        for r in rows:
            w.writerow([r["row_id"], r["lang"], r["kind"], r["question"], r["answer"], source_text(r), ""])
    print(f"wrote {path}: fill keep with y or n for every row, then run freeze")


def adversarial_passages(retriever, question: str, gold_pids: list[str], k: int = 4) -> list[dict]:
    """Frozen passages for one adversarial row: every gold passage, filled with retrieved hits to k.

    Retrieval alone misses the gold passage for many hand-written rows (weak for Hindi), and a
    grounded model must then abstain, so an expected-answer row could never be answered correctly.
    Passages are ordered newest first, as serving orders them, so position does not reveal gold.
    """
    hits = retriever.search(question, k=len(retriever.passages))
    score = {h.pid: h.score for h in hits}
    by_pid = {p["pid"]: p for p in retriever.passages}
    missing = [pid for pid in gold_pids if pid not in by_pid]
    if missing:
        raise SystemExit(f"gold pids not in the index: {missing}")
    chosen = list(dict.fromkeys(gold_pids))
    chosen += [h.pid for h in hits if h.pid not in chosen][: max(0, k - len(chosen))]
    chosen.sort(key=lambda pid: str(by_pid[pid]["date"]), reverse=True)
    return [{"pid": pid, "text": by_pid[pid]["text"], "score": round(score.get(pid, 0.0), 4)} for pid in chosen]


def adversarial_rows(retriever, rows: list[dict], k: int = 4) -> list[dict]:
    out = []
    for i, a in enumerate(rows):
        gold = a.get("gold_pids", []) if a["expected"] == "answer" else []
        out.append({
            "row_id": f"adv-{i:03d}", "lang": a["lang"], "question": a["question"], "kind": "adversarial",
            "passages": adversarial_passages(retriever, a["question"], gold, k),
            "answerable": a["expected"] == "answer", "answer": a.get("gold_answer", ""),
            "cited_pids": gold, "attack_type": a.get("attack_type", ""),
            "must_not_contain": a.get("must_not_contain", []), "split": "test",
        })
    return out


def freeze(index_dir: Path, seed: int, allow_unreviewed: bool) -> None:
    from retrieval.search import Retriever

    rows = read_jsonl(DATA / "curated" / "test.jsonl")
    review = DATA / "test" / "review.csv"
    if review.exists():
        keep = {r["row_id"]: r["keep"].strip().lower() for r in csv.DictReader(review.open(encoding="utf-8-sig"))}
        missing = [rid for rid in (r["row_id"] for r in rows) if keep.get(rid) not in ("y", "n")]
        if missing and not allow_unreviewed:
            raise SystemExit(f"{len(missing)} test rows are not reviewed (keep must be y or n)")
        rows = [r for r in rows if keep.get(r["row_id"]) == "y" or (allow_unreviewed and r["row_id"] in missing)]
    elif not allow_unreviewed:
        raise SystemExit("run `review` and hand-check data/test/review.csv first")

    rng = random.Random(seed)
    answerable = sorted([r for r in rows if r["kind"] == "answerable"], key=lambda r: r["row_id"])
    unanswerable = sorted([r for r in rows if r["kind"] == "unanswerable"], key=lambda r: r["row_id"])
    test = rng.sample(answerable, min(TEST_MIX["answerable"], len(answerable)))
    test += rng.sample(unanswerable, min(TEST_MIX["unanswerable"], len(unanswerable)))

    test += adversarial_rows(Retriever(index_dir), read_jsonl(Path("evals") / "adversarial_test.jsonl"))
    path = DATA / "test" / "test.jsonl"
    write_jsonl(path, test)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (DATA / "test" / "test.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"froze {len(test)} rows: {Counter(r['kind'] for r in test)}; sha256 {digest[:16]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["arms", "review", "freeze"])
    ap.add_argument("--index", type=Path)
    ap.add_argument("--allow-unreviewed", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/arms.yaml").read_text(encoding="utf-8"))
    if a.command == "arms":
        build_arms(cfg["n_rows"], seed=0)
    elif a.command == "review":
        write_review()
    else:
        if not a.index:
            raise SystemExit("--index is required for freeze")
        freeze(a.index, seed=0, allow_unreviewed=a.allow_unreviewed)


if __name__ == "__main__":
    main()
