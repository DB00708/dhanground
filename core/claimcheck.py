"""Deterministic claim check: every number, day count or amount in an answer must appear in a cited passage.

Catches invented numbers only, not wrong reasoning (ED7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PASS, FAIL, NA = "pass", "fail", "n/a"

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_CITATION_RE = re.compile(r"\[[^\[\]]*\]")
_TPLUS_RE = re.compile(r"\bT\s*\+\s*(\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "ninety": 90, "hundred": 100,
}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


@dataclass
class ClaimCheckResult:
    status: str
    claims: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def _norm_number(raw: str) -> str:
    n = raw.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n.lstrip("0") or "0"


def extract_claims(text: str) -> set[str]:
    """Return normalized claims: 'T+N' for settlement windows, plain numbers for everything else."""
    text = _CITATION_RE.sub(" ", text.translate(_DEVA_DIGITS))
    claims: set[str] = set()
    for m in _TPLUS_RE.finditer(text):
        claims.add(f"T+{int(m.group(1))}")
    text = _TPLUS_RE.sub(" ", text)
    for m in _NUMBER_RE.finditer(text):
        claims.add(_norm_number(m.group(0)))
    for m in _WORD_RE.finditer(text):
        claims.add(str(_WORDS[m.group(1).lower()]))
    return claims


def _support_set(passages: list[str]) -> set[str]:
    support: set[str] = set()
    for p in passages:
        claims = extract_claims(p)
        support |= claims
        support |= {c[2:] for c in claims if c.startswith("T+")}
    return support


def check(answer: str, cited_passages: list[str]) -> ClaimCheckResult:
    claims = extract_claims(answer)
    if not claims:
        return ClaimCheckResult(NA)
    support = _support_set(cited_passages)
    unsupported = sorted(c for c in claims if c not in support)
    return ClaimCheckResult(FAIL if unsupported else PASS, sorted(claims), unsupported)
