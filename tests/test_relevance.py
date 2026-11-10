import pytest

from core.relevance import TOC, assign_chapters, chapter_headings, chapter_order, consumer_passages


def doc(*texts, doc_id="md-x"):
    return [{"pid": f"{doc_id}-p{i}", "doc_id": doc_id, "para_id": f"p{i}", "text": t} for i, t in enumerate(texts, 1)]


DOC = doc(
    "Contents Chapter I - Preliminary ............ 1 Chapter III - Digital Lending ............ 9 Chapter XV - Export Credit ............ 80",
    "Chapter I - Preliminary 1. These Directions apply to banks.",
    "Chapter III - Digital Lending 10. A bank shall give a Key Fact Statement to the borrower.",
    "11. Recovery shall follow Chapter XV of these Directions where relevant.",
    "Chapter VIII - Real Estate B. Housing Finance 89. Individual housing loans up to ₹30 lakh: LTV 80%.",
    "C. Loans and advances to Commercial Real Estate 125. CRE exposure limits.",
    "Chapter XV - Export Credit 233. Packing credit period is 360 days.",
    "Interest on export credit shall follow Chapter III - Digital Lending guidance only where applicable.",
)


def test_cross_references_are_not_headings():
    assert chapter_headings("Recovery shall follow Chapter XV of these Directions.") == []
    assert chapter_headings("Chapter XIII A – Credit Facilities to CMIs") == ["XIIIA"]


def test_roman_order():
    assert chapter_order("IV") < chapter_order("V") < chapter_order("VA") < chapter_order("VI") < chapter_order("XIV")


def test_chapters_move_forward_and_skip_contents():
    ch = assign_chapters(DOC)
    assert ch["md-x-p1"] == TOC
    assert [ch[f"md-x-p{i}"] for i in range(2, 9)] == ["I", "III", "III", "VIII", "VIII", "XV", "XV"]


def test_exclusion_with_keep_span_and_terms():
    kept, dropped = consumer_passages(
        DOC, {"md-x": ["VIII", "XV"]}, ["packing credit"],
        {"md-x": [{"from": "B. Housing Finance", "to": "C. Loans and advances to Commercial Real Estate"}]})
    ids = {p["pid"] for p in kept}
    assert ids == {"md-x-p2", "md-x-p3", "md-x-p4", "md-x-p5"}
    assert dropped == {"contents page": 1, "md-x:chapter VIII": 1, "md-x:chapter XV": 2}


def test_missing_keep_span_fails_loudly():
    with pytest.raises(ValueError):
        consumer_passages(DOC, {"md-x": ["VIII"]}, [], {"md-x": [{"from": "No Such Heading", "to": "x"}]})


def test_other_documents_untouched():
    other = doc("UPI: auto reversal latest on T + 1 day.", doc_id="faq-1")
    kept, dropped = consumer_passages(DOC + other, {"md-x": ["XV"]}, [])
    assert "faq-1-p1" in {p["pid"] for p in kept}
