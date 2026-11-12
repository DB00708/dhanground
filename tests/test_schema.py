import json

from core.schema import ABSTAIN_MESSAGES, ModelOutput, parse_output, truncate_words, validate
from core.template import Passage

PASSAGES = [Passage("rbi-11693-p15", "UPI: auto reversal latest on T + 1 day. ₹100/- per day if delay is beyond T + 1 day.")]
Q_LATN = "UPI payment fail hua par paisa kat gaya, kab wapas aayega?"


def out(**kw):
    base = {"answerable": True, "answer": "T + 1 din mein wapas aayega [rbi-11693-p15]",
            "cited_pids": ["rbi-11693-p15"], "lang": "hi-Latn"}
    base.update(kw)
    return ModelOutput(**base)


def test_parse_valid_and_fenced():
    raw = json.dumps(out().model_dump())
    assert parse_output(raw)[0] is not None
    assert parse_output(f"```json\n{raw}\n```")[0] is not None
    assert parse_output(f"Sure! {raw} hope this helps")[0] is not None


def test_parse_errors_give_repair_message():
    obj, err = parse_output("not json")
    assert obj is None and "JSON" in err
    obj, err = parse_output('{"answerable": true, "answer": "x", "cited_pids": [], "lang": "fr"}')
    assert obj is None and "lang" in err
    obj, err = parse_output('{"answerable": true, "answer": "x", "cited_pids": [], "lang": "en", "extra": 1}')
    assert obj is None


def test_happy_path():
    v = validate(out(), PASSAGES, Q_LATN)
    assert v.status == "ok"
    assert v.checks == {"schema": "pass", "citations": "pass", "claim_check": "pass", "lang": "pass"}


def test_unknown_pid_dropped_then_abstain():
    v = validate(out(cited_pids=["rbi-999-p1"]), PASSAGES, Q_LATN)
    assert v.status == "abstained" and v.abstain_reason == "citations"
    assert v.answer == ABSTAIN_MESSAGES["hi-Latn"]


def test_partial_unknown_pid_keeps_known():
    v = validate(out(cited_pids=["rbi-999-p1", "rbi-11693-p15"]), PASSAGES, Q_LATN)
    assert v.status == "ok" and v.cited_pids == ["rbi-11693-p15"] and v.checks["citations"] == "fail"


def test_claim_fail_still_served_with_flag():
    v = validate(out(answer="T + 3 din [rbi-11693-p15]"), PASSAGES, Q_LATN)
    assert v.status == "ok" and v.checks["claim_check"] == "fail" and v.unsupported_claims == ["T+3"]


def test_model_abstention_uses_question_language():
    v = validate(out(answerable=False, cited_pids=[]), PASSAGES, "RBI repo rate kitna hai abhi?")
    assert v.status == "abstained" and v.answer == ABSTAIN_MESSAGES["hi-Latn"]
    v = validate(out(answerable=False, cited_pids=[]), PASSAGES, "यूपीआई रिफंड कब आएगा?")
    assert v.answer == ABSTAIN_MESSAGES["hi-Deva"]


def test_lang_mismatch_is_logged_not_blocked():
    v = validate(out(lang="en"), PASSAGES, Q_LATN)
    assert v.status == "ok" and v.checks["lang"] == "fail"


def test_english_reply_to_unsupported_language_passes_lang_check():
    v = validate(out(lang="en", answer="Reversal by T + 1 day [rbi-11693-p15]"), PASSAGES, "UPI பணம் எப்போது வரும்?")
    assert v.status == "ok" and v.checks["lang"] == "pass"


def test_truncate_at_sentence_boundary():
    text = ("word " * 70).strip() + ". " + ("more " * 70).strip() + "."
    t = truncate_words(text)
    assert t.endswith(".") and len(t.split()) == 70
    assert len(truncate_words("x " * 200).split()) == 120
