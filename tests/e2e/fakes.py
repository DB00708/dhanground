"""Deterministic stand-ins for the teacher LLM and Tiny Aya, used only by the e2e harness.

FakeTeacher answers intelligence.generate prompts; FakeGenerator is selected in serving via
DHANGROUND_GENERATOR=tests.e2e.fakes:FakeGenerator. Both copy passage words (so numbers pass the
claim check), write in the requested language (so the language gates pass), and cite the passage.
"""
from __future__ import annotations

import json
import re

from core import lang as langmod
from intelligence.generate import LANG_DESC

# a-z -> independent Devanagari letters: fake "Hindi" that is all Devanagari script, digits kept.
_DEVA = str.maketrans("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "अबकदएफगहइजकलमनओपकरसतउववकयज" * 2)
_DESC_TO_LANG = {v: k for k, v in LANG_DESC.items()}


def deva(text: str) -> str:
    return text.translate(_DEVA)


def window(text: str, start: int, n: int) -> str:
    words = text.split()
    start = start % max(1, len(words) - n) if len(words) > n else 0
    return " ".join(words[start:start + n])


def question(lang: str, text: str, i: int, answerable: bool) -> str:
    w = window(text, 3 + i * 7, 12)
    if lang == "en":
        return f"What do the RBI rules say about {w}?" if answerable else \
            f"Which bank gives the best rate today for {w}?"
    if lang == "hi-Latn":
        return f"{w} ke baare mein kya niyam hai, mujhe batao?" if answerable else \
            f"{w} ke liye aaj kaun sa bank sabse kam charge leta hai, kya pata hai?"
    return f"{deva(w)} के बारे में नियम क्या है?" if answerable else \
        f"{deva(w)} के लिए आज कौन सा बैंक सबसे कम शुल्क लेता है?"


def answer(lang: str, text: str, pid: str, i: int) -> str:
    w = window(text, i * 7, 30)
    if lang == "en":
        return f"{w} [{pid}]"
    if lang == "hi-Latn":
        return f"RBI ke niyam ke hisaab se yeh hai ki {w} [{pid}]. Aur jaankari ke liye apne bank se baat karein."
    return f"आरबीआई नियम के अनुसार {deva(w)} [{pid}]"


class FakeTeacher:
    """complete(prompt) for intelligence.generate.generate."""

    def __call__(self, prompt: str) -> str:
        n = int(re.search(r"Write (\d+) ", prompt).group(1))
        desc = re.search(r"Language(?: for both question and answer)?: (.+?)\.\n", prompt).group(1)
        lang = _DESC_TO_LANG[desc]
        m = re.search(r"Source passage \[([^\]]+)\] from .*?:\n(.+?)\n\n", prompt, re.S)
        if m:
            pid, text = m.group(1), m.group(2)
            items = [{"question": question(lang, text, i, True), "answer": answer(lang, text, pid, i)}
                     for i in range(n)]
        else:
            text = re.search(r"Topic passage from .*?:\n(.+?)\n\n", prompt, re.S).group(1)
            items = [{"question": question(lang, text, i, False)} for i in range(n)]
        return "```json\n" + json.dumps({"items": items}, ensure_ascii=False) + "\n```"


_Q_RE = re.compile(r"<question>(.*?)</question>", re.S)
_P_RE = re.compile(r"^\[([^\]]+)\] (.*)$", re.M)
_ABSTAIN_RE = re.compile(r"\b(best rate|sabse kam|repo rate|otp|sue)\b|सबसे कम", re.I)


class FakeGenerator:
    """serving.engine.Generator: answers from the first passage, abstains on out-of-scope markers."""

    def __init__(self, base_model: str = "", adapters: dict | None = None):
        self.calls = 0

    def generate(self, messages: list[dict], arm: str, max_time: float = 10.0) -> str:
        self.calls += 1
        user = messages[0]["content"]
        q = _Q_RE.search(user).group(1)
        passages = _P_RE.findall(user.split("<passages>", 1)[1])
        lang = langmod.detect(q)
        lang = lang if lang in LANG_DESC else "en"
        if _ABSTAIN_RE.search(q) or not passages:
            return json.dumps({"answerable": False, "answer": "", "cited_pids": [], "lang": lang})
        pid, text = passages[0]
        return json.dumps({"answerable": True, "answer": answer(lang, text, pid, 0),
                           "cited_pids": [pid], "lang": lang}, ensure_ascii=False)
