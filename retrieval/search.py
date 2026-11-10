"""Hybrid BM25 + multilingual-embedding retrieval over the paragraph corpus (component 7, ED6).

The ranking score is a weighted sum of per-query min-max normalized scores. The gate score is
absolute (embedding cosine when an embedding index exists, else raw BM25) so tau means the same
thing for every question.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_TPLUS_RE = re.compile(r"\bt\s*\+\s*(\d+)")
_TOKEN_RE = re.compile(r"t\+\d+|[\wऀ-ॣ०-ॿ]+", re.UNICODE)  # \w alone splits Hindi at vowel signs

# Hinglish/Hindi terms -> English corpus vocabulary. Lets BM25 match romanized questions.
HINGLISH_TERMS = {
    "paisa": "money amount", "paise": "money amount", "rupaye": "rupees amount", "rupay": "rupees",
    "kat": "debited", "kata": "debited", "kaat": "debited", "katna": "debited",
    "wapas": "refund reversal reversed", "wapis": "refund reversal reversed", "wpas": "refund reversal",
    "lautana": "refund reversal", "khata": "account", "khaate": "account", "khate": "account",
    "shikayat": "complaint grievance", "shikaayat": "complaint grievance", "din": "days day",
    "dino": "days", "bheja": "transfer sent remitter", "bheje": "transfer sent", "mila": "credited received",
    "nahi": "not", "nikla": "dispensed withdrawal", "nikle": "dispensed withdrawal",
    "dhokha": "fraud", "dhokhadhadi": "fraud", "fraud": "fraud unauthorised", "jurmana": "penalty compensation",
    "muavza": "compensation", "byaj": "interest", "karz": "loan", "loan": "loan credit", "kist": "emi instalment",
    "lokpal": "ombudsman", "naam": "name", "pata": "address", "pehchan": "identity kyc",
    "पैसा": "money amount", "पैसे": "money amount", "खाता": "account", "खाते": "account",
    "वापस": "refund reversal", "शिकायत": "complaint grievance", "दिन": "days day", "कटा": "debited",
    "कट": "debited", "लोकपाल": "ombudsman", "धोखाधड़ी": "fraud", "ब्याज": "interest", "ऋण": "loan",
    "मुआवजा": "compensation", "निकला": "dispensed withdrawal", "आरबीआई": "rbi", "यूपीआई": "upi",
    "एटीएम": "atm", "केवाईसी": "kyc", "बैंक": "bank", "दर्ज": "lodge file", "बीमा": "insurance",
}


def tokenize(text: str) -> list[str]:
    text = _TPLUS_RE.sub(lambda m: f"t+{m.group(1)}", text.lower().translate(_DEVA_DIGITS))
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        tokens.append(tok)
        if tok in HINGLISH_TERMS:
            tokens.extend(HINGLISH_TERMS[tok].split())
    return tokens


@dataclass
class Hit:
    pid: str
    text: str
    score: float
    gate_score: float
    doc_id: str
    title: str
    url: str
    date: str


class Retriever:
    def __init__(self, index_dir: Path, bm25_weight: float = 0.5, embedding_weight: float = 0.5):
        index_dir = Path(index_dir)
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.passages = [json.loads(l) for l in (index_dir / "passages.jsonl").open(encoding="utf-8")]
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in self.passages])
        self.bm25_weight, self.embedding_weight = bm25_weight, embedding_weight
        self.encoder = None
        self.embeddings = None
        emb_path = index_dir / "embeddings.npy"
        if self.meta.get("embedding_model") and emb_path.exists():
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.meta["embedding_model"])
            self.embeddings = np.load(emb_path)

    @property
    def index_version(self) -> str:
        return self.meta["index_version"]

    def _embed_query(self, question: str) -> np.ndarray:
        prefix = self.meta.get("query_prefix", "")
        return self.encoder.encode([prefix + question], normalize_embeddings=True)[0]

    def search(self, question: str, k: int = 4, exclude_docs: set[str] | None = None) -> list[Hit]:
        bm = np.asarray(self.bm25.get_scores(tokenize(question)), dtype=float)
        if self.embeddings is not None:
            cos = self.embeddings @ self._embed_query(question)
            fused = self.bm25_weight * _minmax(bm) + self.embedding_weight * _minmax(cos)
            gate = cos
        else:
            fused, gate = _minmax(bm), bm
        order = np.argsort(-fused)
        hits = []
        for i in order:
            p = self.passages[i]
            if exclude_docs and p["doc_id"] in exclude_docs:
                continue
            hits.append(Hit(p["pid"], p["text"], float(fused[i]), float(gate[i]), p["doc_id"],
                            p["title"], p["url"], str(p["date"])))
            if len(hits) == k:
                break
        # Newer documents first, so an amended rule is read before the one it replaced.
        return sorted(hits, key=lambda h: h.date, reverse=True)

    def top_gate_score(self, hits: list[Hit]) -> float:
        return max((h.gate_score for h in hits), default=0.0)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)
