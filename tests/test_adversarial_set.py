"""The hand-written adversarial test set (evals/adversarial_test.jsonl) that build_arms.freeze() reads."""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from core import claimcheck
from core.lang import detect

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "evals" / "adversarial_test.jsonl"
ATTACKS = {"injection", "out_of_corpus", "scam_impersonation", "fraud_enabling", "personal_advice",
           "typo_mixed_script"}
LANGS = {"hi-Deva", "hi-Latn", "en"}
REQUIRED = {"question", "lang", "expected", "attack_type"}
OPTIONAL = {"gold_answer", "gold_pids", "status", "must_not_contain"}


def _corpus_dir() -> Path | None:
    # The corpus is git-ignored and lives in the main checkout; a worktree sits under .claude/worktrees.
    for base in (ROOT, *ROOT.parents):
        if (base / "corpus").is_dir() and any((base / "corpus").glob("*.jsonl")):
            return base / "corpus"
    return None


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line) for line in SET.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    d = _corpus_dir()
    if d is None:
        pytest.skip("corpus/ not present (git-ignored; built in the main checkout)")
    out = {}
    for f in d.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                out[p["pid"]] = p["text"]
    return out


def test_exactly_100_rows(rows):
    assert len(rows) == 100


def test_fields_are_valid(rows):
    for i, r in enumerate(rows):
        assert REQUIRED <= r.keys() and r.keys() <= REQUIRED | OPTIONAL, (i, sorted(r))
        assert isinstance(r["question"], str) and r["question"].strip() == r["question"] and r["question"], i
        assert r["lang"] in LANGS and r["attack_type"] in ATTACKS, i
        assert r["expected"] in ("answer", "abstain"), i
        if r["expected"] == "answer":
            assert isinstance(r.get("gold_answer"), str) and r["gold_answer"].strip(), i
            assert isinstance(r.get("gold_pids"), list) and r["gold_pids"], i
            assert len(set(r["gold_pids"])) == len(r["gold_pids"]), i
        else:
            assert "gold_answer" not in r and "gold_pids" not in r, i
        assert r.get("status") in (None, "draft", "reviewed"), i  # seed rows carry no status


def test_no_duplicate_questions(rows):
    norm = [" ".join(r["question"].casefold().split()) for r in rows]
    dupes = [q for q, n in Counter(norm).items() if n > 1]
    assert not dupes, dupes


def test_lang_labels_match_detector(rows):
    wrong = [(i, r["lang"], detect(r["question"])) for i, r in enumerate(rows) if detect(r["question"]) != r["lang"]]
    assert not wrong, wrong


def test_mix_matches_the_design(rows):
    attacks, langs = Counter(r["attack_type"] for r in rows), Counter(r["lang"] for r in rows)
    assert attacks == {"injection": 30, "out_of_corpus": 20, "scam_impersonation": 15, "fraud_enabling": 15,
                       "personal_advice": 10, "typo_mixed_script": 10}
    assert langs == {"hi-Latn": 40, "hi-Deva": 30, "en": 30}
    assert all(r["expected"] == "abstain" for r in rows if r["attack_type"] in ("out_of_corpus", "fraud_enabling"))


def test_gold_pids_exist_in_corpus(rows, corpus):
    missing = [(i, p) for i, r in enumerate(rows) for p in r.get("gold_pids", []) if p not in corpus]
    assert not missing, missing


def test_gold_answers_invent_no_numbers(rows, corpus):
    # run_eval fails an answer whose numbers are not in its cited passages; gold must pass that bar.
    bad = []
    for i, r in enumerate(rows):
        if r["expected"] == "answer":
            res = claimcheck.check(r["gold_answer"], [corpus[p] for p in r["gold_pids"]])
            if res.status == claimcheck.FAIL:
                bad.append((i, res.unsupported))
    assert not bad, bad


def test_must_not_contain_is_valid_and_spares_the_gold_answer(rows):
    for i, r in enumerate(rows):
        pats = r.get("must_not_contain")
        if pats is None:
            continue
        assert r["expected"] == "answer" and isinstance(pats, list) and pats, i
        for rx in pats:
            re.compile(rx)
            assert not re.search(rx, r["gold_answer"], re.IGNORECASE), (i, rx)


def test_planted_number_injections_list_their_claim(rows):
    # Injections whose planted claim would pass the claim check (no number, or a supported one) need a
    # must_not_contain entry; rows with no planted claim at all are left to the judge.
    with_claim = [r for r in rows if r["attack_type"] == "injection" and r["expected"] == "answer"
                  and r.get("must_not_contain")]
    assert len(with_claim) >= 20
