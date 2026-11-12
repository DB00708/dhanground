"""Keep question generation on retail-customer content.

Master Directions are long and mix consumer rules with corporate ones (export credit, project finance).
Passages are tagged with the chapter they sit in, and chapters listed in config are skipped for
generation. Retrieval still indexes everything, so off-topic questions are answered or abstained honestly.
"""
from __future__ import annotations

import re

_HEADING_RE = re.compile(r"Chapter\s+([IVXL]+)\s*(A)?\s*[-–:]\s*(?!\s)")
# "Chapter XI – Acquisition Finance of these Directions" is a cross-reference; "Chapter II - Role of the Board" is a heading.
_REFERENCE_RE = re.compile(r"^[^.]{0,80}?\bof (?:these|the) Directions\b")


def chapter_headings(text: str) -> list[str]:
    """Chapter numbers that start a chapter in this text; cross-references ('Chapter XV of these Directions') are ignored."""
    found = []
    for m in _HEADING_RE.finditer(text):
        if _REFERENCE_RE.match(text[m.end():m.end() + 120]):
            continue
        label = m.group(1) + (m.group(2) or "")
        if label not in found:
            found.append(label)
    return found


TOC = "toc"
_TOC_RE = re.compile(r"\.{10,}")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def chapter_order(label: str) -> float:
    numeral, suffix = label.rstrip("A"), label.endswith("A")
    total = 0
    for i, ch in enumerate(numeral):
        v = _ROMAN[ch]
        total += -v if i + 1 < len(numeral) and _ROMAN[numeral[i + 1]] > v else v
    return total + (0.5 if suffix else 0)


def assign_chapters(passages: list[dict]) -> dict[str, str | None]:
    """pid -> chapter label, walking a document's passages in order.

    Contents pages (dotted leader lines) are tagged 'toc' and do not move the chapter. Chapters only
    move forward, so a cross-reference to an earlier chapter is not mistaken for a heading.
    """
    current = None
    out = {}
    for p in sorted(passages, key=lambda p: int(p["para_id"].lstrip("p"))):
        if _TOC_RE.search(p["text"]):
            out[p["pid"]] = TOC
            continue
        for head in chapter_headings(p["text"]):
            if current is None or chapter_order(head) > chapter_order(current):
                current = head
        out[p["pid"]] = current
    return out


def excluded_term(text: str, terms: list[str]) -> str | None:
    low = text.lower()
    return next((t for t in terms if t.lower() in low), None)


def span_pids(passages: list[dict], start: str, end: str | None) -> set[str]:
    """pids from the passage containing `start` up to (not including) the one containing `end`, case-insensitive.

    `end` None runs to the end of the document. Skips contents pages, which also contain the heading text.
    """
    inside, out = False, set()
    for p in sorted(passages, key=lambda p: int(p["para_id"].lstrip("p"))):
        low = p["text"].lower()
        if _TOC_RE.search(p["text"]):
            continue
        if inside and end is not None and end.lower() in low:
            break
        if start.lower() in low:
            inside = True
        if inside:
            out.add(p["pid"])
    return out


def _spans(passages: list[dict], doc_id: str, spans: list[dict], kind: str) -> set[str]:
    out = set()
    for span in spans:
        found = span_pids(passages, span["from"], span.get("to"))
        if not found:
            raise ValueError(f"{kind} span not found in {doc_id}: {span['from']!r}")
        out |= found
    return out


def consumer_passages(passages: list[dict], exclude_chapters: dict[str, list[str]],
                      exclude_terms: list[str], keep_spans: dict[str, list[dict]] | None = None,
                      exclude_spans: dict[str, list[dict]] | None = None,
                      ) -> tuple[list[dict], dict[str, int]]:
    """Returns (kept passages, drop counts by reason).

    keep_spans ({doc: [{from, to}]}) overrides chapter exclusion. exclude_spans (same shape) drops sections
    inside a kept chapter, or in a document without chapter headings; it wins over keep_spans. A span `to`
    of None runs to the end of the document. A span whose `from` text is not found raises ValueError.
    """
    by_doc: dict[str, list[dict]] = {}
    for p in passages:
        by_doc.setdefault(p["doc_id"], []).append(p)
    kept, dropped = [], {}
    for doc_id, ps in by_doc.items():
        skip = set(exclude_chapters.get(doc_id, [])) | {TOC}
        chapters = assign_chapters(ps)
        keep = _spans(ps, doc_id, (keep_spans or {}).get(doc_id, []), "keep")
        drop = _spans(ps, doc_id, (exclude_spans or {}).get(doc_id, []), "exclude")
        for p in ps:
            reason = None
            if chapters.get(p["pid"]) == TOC:
                reason = "contents page"
            elif p["pid"] in drop:
                reason = f"{doc_id}:span"
            elif chapters.get(p["pid"]) in skip and p["pid"] not in keep:
                reason = f"{doc_id}:chapter {chapters[p['pid']]}"
            elif term := excluded_term(p["text"], exclude_terms):
                reason = f"term:{term}"
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                kept.append(p)
    return kept, dropped
