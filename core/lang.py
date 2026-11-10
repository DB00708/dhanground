"""Language and script detection for hi-Deva, hi-Latn and en; anything else is 'other'."""
from __future__ import annotations

import re

HI_DEVA, HI_LATN, EN, OTHER = "hi-Deva", "hi-Latn", "en", "other"

_WORD_RE = re.compile(r"\w+", re.UNICODE)

_HINGLISH = {
    "hai", "hain", "kya", "kab", "kaise", "kaun", "kyun", "kyon", "nahi", "nahin", "mera", "meri",
    "mere", "mujhe", "maine", "humne", "paisa", "paise", "wapas", "wapis", "wpas", "aayega", "ayega",
    "aaega", "karun", "karu", "karna", "karein", "kare", "ka", "ki", "ke", "ko", "se", "mein", "me",
    "hua", "hui", "gaya", "gayi", "kat", "kata", "abhi", "kitna", "kitne", "kitni", "din", "bhi",
    "aur", "toh", "tha", "thi", "hoga", "hogi", "chahiye", "sakta", "sakte", "sakti", "bheja",
    "mila", "nikla", "nikle", "agar", "lekin", "kuch", "koi", "apna", "apne", "batao", "bataiye",
    "jaankari", "pata", "wala", "wali", "raha", "rahi", "rahe", "diya", "liya", "kaha",
}


def _script(word: str) -> str:
    for ch in word:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "deva"
        if ch.isascii() and ch.isalpha():
            return "latn"
        if ch.isalpha():
            return "other"
    return "none"


def detect(text: str) -> str:
    words = _WORD_RE.findall(text)
    counts = {"deva": 0, "latn": 0, "other": 0}
    latin_words = []
    for w in words:
        s = _script(w)
        if s in counts:
            counts[s] += 1
            if s == "latn":
                latin_words.append(w.lower())
    if not any(counts.values()):
        return OTHER
    top = max(counts, key=counts.get)
    if top == "other":
        return OTHER
    if top == "deva":
        return HI_DEVA
    hits = sum(1 for w in latin_words if w in _HINGLISH)
    if hits >= 2 or (hits >= 1 and hits / len(latin_words) >= 0.15):
        return HI_LATN
    return EN
