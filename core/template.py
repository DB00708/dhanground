"""Prompt template. Identical for every arm, at train and serve time.

All instructions sit in the user turn because AlignTune SFT drops the system turn.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

INSTRUCTIONS = (
    "You answer Indian banking questions using ONLY the passages.\n"
    "Reply in the same language and script as the question.\n"
    "Cite passage ids in square brackets.\n"
    "If the passages do not answer it, set answerable=false.\n"
    "Return one JSON object and nothing else."
)

MAX_PASSAGES = 4

_TAG_RE = re.compile(r"</?\s*(question|passages)\s*>", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{0,40}\|>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Passage:
    pid: str
    text: str
    score: float = 0.0


def sanitize(text: str) -> str:
    """Untrusted text can never close our tags or inject chat-template tokens."""
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def render_user(question: str, passages: list[Passage]) -> str:
    if not 1 <= len(passages) <= MAX_PASSAGES:
        raise ValueError(f"expected 1..{MAX_PASSAGES} passages, got {len(passages)}")
    lines = [f"[{p.pid}] {sanitize(p.text)}" for p in passages]
    return (
        f"{INSTRUCTIONS}\n\n"
        f"<question>{sanitize(question)}</question>\n"
        "<passages>\n" + "\n".join(lines) + "\n</passages>"
    )


def render_assistant(answerable: bool, answer: str, cited_pids: list[str], lang: str) -> str:
    return json.dumps(
        {"answerable": answerable, "answer": answer, "cited_pids": cited_pids, "lang": lang},
        ensure_ascii=False,
    )


def to_sft_messages(question: str, passages: list[Passage], output: dict) -> list[dict]:
    return [
        {"role": "user", "content": render_user(question, passages)},
        {
            "role": "assistant",
            "content": render_assistant(
                output["answerable"], output["answer"], output["cited_pids"], output["lang"]
            ),
        },
    ]
