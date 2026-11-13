"""Model output schema, training row schema, and backend validation V1..V5 (BE <> AI contract)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from core import claimcheck, lang as langmod
from core.template import Passage

Lang = Literal["hi-Latn", "hi-Deva", "en"]
MAX_WORDS = 120

ABSTAIN_MESSAGES = {
    "hi-Latn": (
        "Yeh jaankari mere sources mein nahi hai. Apne bank se sampark karein, ya RBI Integrated "
        "Ombudsman (cms.rbi.org.in) par shikayat darj karein."
    ),
    "hi-Deva": (
        "यह जानकारी मेरे स्रोतों में नहीं है। अपने बैंक से संपर्क करें, या RBI एकीकृत लोकपाल "
        "(cms.rbi.org.in) पर शिकायत दर्ज करें।"
    ),
    "en": (
        "This information is not in my sources. Please contact your bank, or file a complaint "
        "with the RBI Integrated Ombudsman (cms.rbi.org.in)."
    ),
}


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: bool
    answer: str
    cited_pids: list[str]
    lang: Lang


class Gates(BaseModel):
    dedup: str
    quality: float
    halluc: str
    provenance: str


class Provenance(BaseModel):
    source_doc: str
    span_sha256: str
    span_sha256s: list[str] | None = None   # multi-passage rows: one hash per source_pids entry
    generator: str
    gates: Gates | None = None


class RowPassage(BaseModel):
    pid: str
    text: str
    score: float = 0.0


class TrainingRow(BaseModel):
    row_id: str
    lang: Lang
    question: str
    passages: list[RowPassage]
    answerable: bool
    answer: str
    cited_pids: list[str]
    numbers: list[str] = []
    provenance: Provenance
    split: Literal["train", "dev", "test"]
    kind: Literal["answerable", "unanswerable", "adversarial"] = "answerable"
    source_pid: str | None = None
    source_pids: list[str] | None = None   # multi-passage rows: every gold passage; source_pid is the first
    noise: Literal["typo"] | None = None


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_output(raw: str) -> tuple[ModelOutput | None, str | None]:
    """V1. Returns (output, None) or (None, error message for the single repair retry)."""
    text = _FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, "Output is not JSON. Return one JSON object and nothing else."
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e.msg}. Return one JSON object and nothing else."
    try:
        return ModelOutput.model_validate(data), None
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'root'}: {err['msg']}" for err in e.errors())
        return None, f"JSON does not match the schema: {problems}"


@dataclass
class Validated:
    status: Literal["ok", "abstained"]
    answerable: bool
    answer: str
    lang: str
    cited_pids: list[str]
    checks: dict = field(default_factory=dict)
    abstain_reason: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)


def abstain(lang: str, reason: str, checks: dict | None = None) -> Validated:
    lang = lang if lang in ABSTAIN_MESSAGES else "en"
    return Validated("abstained", False, ABSTAIN_MESSAGES[lang], lang, [], checks or {}, reason)


def truncate_words(text: str, limit: int = MAX_WORDS) -> str:
    """V5. Cut at a sentence boundary; hard-cut words only if the first sentence is over the limit."""
    if len(text.split()) <= limit:
        return text
    kept: list[str] = []
    count = 0
    for sentence in re.split(r"(?<=[.!?।])\s+", text):
        n = len(sentence.split())
        if count + n > limit:
            break
        kept.append(sentence)
        count += n
    return " ".join(kept) if kept else " ".join(text.split()[:limit])


def validate(output: ModelOutput, passages: list[Passage], question: str,
             question_lang: str | None = None) -> Validated:
    """V2..V5 on a schema-valid output. V1 (parse + repair retry) happens in the answer service.

    Callers that already ran langmod.detect(question) pass question_lang to skip a second detect."""
    if question_lang is None:
        question_lang = langmod.detect(question)
    reply_lang = langmod.reply_lang(question_lang)
    checks = {"schema": "pass"}

    if not output.answerable:
        checks.update(citations="n/a", claim_check="n/a")
        return abstain(reply_lang, "model", checks)

    by_pid = {p.pid: p for p in passages}
    cited = list(dict.fromkeys(pid for pid in output.cited_pids if pid in by_pid))
    checks["citations"] = "pass" if len(cited) == len(output.cited_pids) and cited else "fail"
    if not cited:
        checks["claim_check"] = "n/a"
        return abstain(reply_lang, "citations", checks)

    answer = truncate_words(output.answer)
    cc = claimcheck.check(answer, [by_pid[pid].text for pid in cited])
    checks["claim_check"] = cc.status
    checks["lang"] = "pass" if output.lang == reply_lang else "fail"

    return Validated("ok", True, answer, output.lang, cited, checks, None, cc.unsupported)
