"""Mask account and card numbers before a question leaves the FE or reaches a log."""
from __future__ import annotations

import re

MASK = "[MASKED]"
LOG_MAX_CHARS = 200

_D = "[0-9०-९]"
_CARD_RE = re.compile(rf"(?<!{_D})(?:{_D}{{4}}[ -]){{2,3}}{_D}{{4}}(?!{_D})")
_LONG_RUN_RE = re.compile(rf"(?<!{_D}){_D}{{9,18}}(?!{_D})")


def mask(text: str) -> tuple[str, int]:
    text, n_card = _CARD_RE.subn(MASK, text)
    text, n_run = _LONG_RUN_RE.subn(MASK, text)
    return text, n_card + n_run


def for_log(text: str) -> str:
    return mask(text)[0][:LOG_MAX_CHARS]
