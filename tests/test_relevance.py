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


def test_heading_with_of_the_in_title_is_not_a_reference():
    assert chapter_headings("e may be. Chapter II - Role of the Board A. Board Approved Policies 7.") == ["II"]
    assert chapter_headings("see Chapter XI – Acquisition Finance of these Directions.") == []


def test_exclude_spans_drop_sections_and_open_ended_span_runs_to_end():
    flat = doc("1. Definitions. A PPI is a prepaid instrument.", "3. Eligibility for issuers. Net worth.",
               "4. Capital.", "6. Safeguards. KYC applies to holders.", "18. Information system audit.",
               "Annex escrow report format.", doc_id="md-y")
    kept, dropped = consumer_passages(flat, {}, [], None, {"md-y": [
        {"from": "3. Eligibility", "to": "6. Safeguards"}, {"from": "18. Information system audit", "to": None}]})
    assert {p["pid"] for p in kept} == {"md-y-p1", "md-y-p4"}
    assert dropped == {"md-y:span": 4}


def test_exclude_span_wins_over_keep_span_and_missing_one_fails():
    kept, _ = consumer_passages(
        DOC, {"md-x": ["VIII"]}, [],
        {"md-x": [{"from": "B. Housing Finance", "to": "C. Loans and advances"}]},
        {"md-x": [{"from": "B. Housing Finance", "to": "C. Loans and advances"}]})
    assert "md-x-p5" not in {p["pid"] for p in kept}
    with pytest.raises(ValueError):
        consumer_passages(DOC, {}, [], None, {"md-x": [{"from": "No Such Heading", "to": None}]})


def test_configured_spans_resolve_on_the_real_index():
    """Every span in configs/generation.yaml must match text in the built index.

    Uses ./index, or the index directory in DHANGROUND_INDEX_DIR; skipped where neither exists.
    """
    import json
    import os
    from pathlib import Path

    import yaml
    from core.relevance import span_pids

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.get("DHANGROUND_INDEX_DIR")
    found = [Path(env) / "passages.jsonl"] if env else sorted((repo / "index").glob("*/passages.jsonl"))
    if not found or not found[-1].exists():
        pytest.skip("no built index in this checkout")
    cfg = yaml.safe_load((repo / "configs" / "generation.yaml").read_text(encoding="utf-8"))
    passages = [json.loads(l) for l in found[-1].open(encoding="utf-8")]
    for kind in ("keep_spans", "exclude_spans"):
        for doc_id, spans in (cfg.get(kind) or {}).items():
            ps = [p for p in passages if p["doc_id"] == doc_id]
            for span in spans:
                assert span_pids(ps, span["from"], span.get("to")), (kind, doc_id, span["from"])
