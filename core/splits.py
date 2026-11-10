"""Split by source document (never by row) and fail the build on any train/test overlap (ED9)."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

SPLITS = ("train", "dev", "test")


class LeakageError(Exception):
    pass


def _key(seed: int, doc_id: str) -> str:
    return hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()


def split_docs(doc_ids: list[str], ratios: dict[str, float], seed: int = 0) -> dict[str, list[str]]:
    """Deterministic: same doc ids + seed always give the same split, regardless of input order."""
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    ordered = sorted(set(doc_ids), key=lambda d: _key(seed, d))
    n = len(ordered)
    n_dev = max(1, round(n * ratios["dev"])) if ratios["dev"] > 0 else 0
    n_test = max(1, round(n * ratios["test"])) if ratios["test"] > 0 else 0
    if n_dev + n_test >= n:
        raise ValueError(f"{n} documents is too few for ratios {ratios}")
    return {
        "test": sorted(ordered[:n_test]),
        "dev": sorted(ordered[n_test : n_test + n_dev]),
        "train": sorted(ordered[n_test + n_dev :]),
    }


def split_docs_stratified(
    doc_topics: dict[str, list[str]],
    doc_sizes: dict[str, int],
    core_topics: list[str],
    ratios: dict[str, float],
    seed: int = 0,
) -> dict[str, list[str]]:
    """Split by document, sized by passages, so held-out splits cover the core topics.

    1. Test, then dev: for each core topic not yet covered, move in one document carrying it,
       but only if another document with that topic stays in train and the document fits the
       split's passage budget. Rarest topics are placed first; the smallest eligible document wins.
    2. Top up test, then dev, toward their passage budgets with documents that still fit.
    Deterministic for a given input and seed.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    total = sum(doc_sizes[d] for d in doc_topics)
    train = set(doc_topics)
    held: dict[str, set[str]] = {"test": set(), "dev": set()}
    order = sorted(doc_topics, key=lambda d: _key(seed, d))

    def size(split: str) -> int:
        return sum(doc_sizes[d] for d in held[split])

    def fits(split: str, d: str) -> bool:
        return size(split) + doc_sizes[d] <= ratios[split] * total

    def train_keeps(d: str) -> bool:
        return all(any(t in doc_topics[o] for o in train - {d}) for t in doc_topics[d] if t in core_topics)

    rarity = sorted(core_topics, key=lambda t: (sum(t in ts for ts in doc_topics.values()), t))
    for split in ("test", "dev"):
        for topic in rarity:
            if any(topic in doc_topics[d] for d in held[split]):
                continue
            eligible = [d for d in order if d in train and topic in doc_topics[d] and fits(split, d) and train_keeps(d)]
            if eligible:
                d = min(eligible, key=lambda x: doc_sizes[x])  # smallest first leaves room for other topics
                train.discard(d)
                held[split].add(d)
        for d in order:
            if d in train and fits(split, d) and train_keeps(d) and size(split) < ratios[split] * total:
                train.discard(d)
                held[split].add(d)
    if not held["test"] or not held["dev"]:
        raise ValueError("too few documents to fill test and dev")
    return {"test": sorted(held["test"]), "dev": sorted(held["dev"]), "train": sorted(train)}


def normalize_question(q: str) -> str:
    return re.sub(r"[^\w]+", " ", q.lower()).strip()


def check_no_overlap(rows: list[dict]) -> None:
    """Rows need: split, question, provenance.source_doc, provenance.span_sha256."""
    seen: dict[str, dict[str, set[str]]] = {
        "doc": defaultdict(set), "span": defaultdict(set), "question": defaultdict(set)
    }
    for r in rows:
        prov = r["provenance"]
        seen["doc"][prov["source_doc"]].add(r["split"])
        seen["span"][prov["span_sha256"]].add(r["split"])
        qh = hashlib.sha256(normalize_question(r["question"]).encode()).hexdigest()
        seen["question"][qh].add(r["split"])

    problems = []
    for kind, index in seen.items():
        leaked = [k for k, splits in index.items() if len(splits) > 1]
        if leaked:
            problems.append(f"{len(leaked)} {kind} value(s) in more than one split, e.g. {leaked[0][:16]}")
    if problems:
        raise LeakageError("; ".join(problems))
