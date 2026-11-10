import pytest

from core.splits import LeakageError, check_no_overlap, split_docs

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
