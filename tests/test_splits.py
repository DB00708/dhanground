import pytest

from core.splits import LeakageError, check_no_overlap, split_docs, split_docs_stratified

CORE = ["upi", "atm", "kyc", "fraud"]
TOPICS = {
    "tat": ["upi", "atm"], "ppi": ["upi", "fraud"], "mandate": ["upi"], "atm-faq": ["atm"],
    "kyc-md": ["kyc"], "kyc-faq": ["kyc"], "fraud-md": ["fraud"], "huge": ["fraud", "kyc"],
    "loan": ["loans"], "currency": ["currency"], "neft": ["neft"], "deposit": ["deposits"],
}
SIZES = {"tat": 27, "ppi": 212, "mandate": 15, "atm-faq": 32, "kyc-md": 404, "kyc-faq": 41,
         "fraud-md": 107, "huge": 619, "loan": 9, "currency": 90, "neft": 34, "deposit": 25}
PR = {"train": 0.76, "dev": 0.08, "test": 0.16}


def test_stratified_test_covers_core_topics_and_train_keeps_them():
    s = split_docs_stratified(TOPICS, SIZES, CORE, PR)
    for split in ("train", "test"):
        assert {t for d in s[split] for t in TOPICS[d]} >= set(CORE), split


def test_stratified_respects_passage_budget_and_partitions():
    s = split_docs_stratified(TOPICS, SIZES, CORE, PR)
    total = sum(SIZES.values())
    for split in ("test", "dev"):
        assert sum(SIZES[d] for d in s[split]) <= PR[split] * total
    assert "huge" in s["train"] and "kyc-md" in s["train"]
    assert sorted(s["train"] + s["dev"] + s["test"]) == sorted(TOPICS)


def test_stratified_never_moves_a_topics_only_doc():
    s = split_docs_stratified({"only-upi": ["upi"], "a": ["x"], "b": ["y"], "c": ["z"]},
                              {"only-upi": 5, "a": 5, "b": 5, "c": 5}, ["upi"], {"train": 0.5, "dev": 0.25, "test": 0.25})
    assert "only-upi" in s["train"]


def test_stratified_is_deterministic():
    assert split_docs_stratified(TOPICS, SIZES, CORE, PR) == split_docs_stratified(dict(reversed(TOPICS.items())), SIZES, CORE, PR)

DOCS = [f"rbi-{i}" for i in range(50)]
RATIOS = {"train": 0.76, "dev": 0.08, "test": 0.16}


def test_split_is_deterministic_and_order_independent():
    a = split_docs(DOCS, RATIOS, seed=0)
    b = split_docs(list(reversed(DOCS)), RATIOS, seed=0)
    assert a == b
    assert split_docs(DOCS, RATIOS, seed=1) != a


def test_split_partitions_all_docs():
    s = split_docs(DOCS, RATIOS)
    all_docs = s["train"] + s["dev"] + s["test"]
    assert sorted(all_docs) == sorted(DOCS) and len(set(all_docs)) == len(DOCS)
    assert len(s["test"]) == 8 and len(s["dev"]) == 4


def test_too_few_docs_rejected():
    with pytest.raises(ValueError):
        split_docs(["a", "b"], RATIOS)


def row(split, doc, span, q):
    return {"split": split, "question": q, "provenance": {"source_doc": doc, "span_sha256": span}}


def test_clean_rows_pass():
    check_no_overlap([row("train", "d1", "s1", "q one"), row("test", "d2", "s2", "q two")])


@pytest.mark.parametrize("rows", [
    [row("train", "d1", "s1", "q1"), row("test", "d1", "s2", "q2")],
    [row("train", "d1", "s1", "q1"), row("test", "d2", "s1", "q2")],
    [row("train", "d1", "s1", "UPI refund kab?"), row("test", "d2", "s2", "upi refund KAB")],
])
def test_leakage_fails_build(rows):
    with pytest.raises(LeakageError):
        check_no_overlap(rows)
