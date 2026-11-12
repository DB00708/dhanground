import json

from core.schema import ABSTAIN_MESSAGES
from retrieval.search import Retriever, tokenize
from serving.engine import AnswerEngine, TimeoutGeneration


def test_tokenize_joins_t_plus_and_expands_hinglish():
    toks = tokenize("Paisa kat gaya, T + 1 din")
    assert "t+1" in toks and "debited" in toks and "days" in toks


def test_devanagari_words_stay_whole():
    toks = tokenize("आरबीआई लोकपाल में शिकायत")
    assert "शिकायत" in toks and "complaint" in toks and "ombudsman" in toks and "rbi" in toks


def test_devanagari_question_finds_ombudsman(index_dir):
    hits = Retriever(index_dir).search("आरबीआई लोकपाल में शिकायत कैसे दर्ज करें?", k=3)
    assert max(hits, key=lambda h: h.score).pid == "faq-153-p4"


def test_hinglish_question_finds_upi_rule(index_dir):
    hits = Retriever(index_dir).search("UPI payment fail, paisa kat gaya, kab wapas aayega?", k=1)
    assert hits[0].pid == "rbi-11693-p15"


def test_exclude_docs_and_newer_first(index_dir):
    r = Retriever(index_dir)
    assert all(h.doc_id != "rbi-11693" for h in r.search("ombudsman complaint", k=3, exclude_docs={"rbi-11693"}))
    hits = r.search("complaint refund days", k=3)
    assert [h.date for h in hits] == sorted([h.date for h in hits], reverse=True)


class FakeGen:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, messages, arm, max_time):
        self.calls.append((messages, arm))
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


GOOD = json.dumps({"answerable": True, "answer": "T + 1 din mein paisa wapas aayega [rbi-11693-p15]",
                   "cited_pids": ["rbi-11693-p15"], "lang": "hi-Latn"})
Q = "UPI payment fail hua, paisa kat gaya, kab wapas aayega?"


def test_happy_path_returns_contract_fields(index_dir):
    eng = AnswerEngine(FakeGen([GOOD]), Retriever(index_dir), tau=None)
    r = eng.answer(Q, "A3", request_id="r1").response
    assert r["status"] == "ok" and r["checks"]["claim_check"] == "pass"
    assert r["citations"][0]["pid"] == "rbi-11693-p15" and r["citations"][0]["url"]
    assert r["index_version"] == "v1" and r["source"] == "live"


def test_one_repair_then_abstain(index_dir):
    gen = FakeGen(["not json", "still not json"])
    r = AnswerEngine(gen, Retriever(index_dir)).answer(Q, "A1").response
    assert len(gen.calls) == 2 and "previous reply was not valid" in gen.calls[1][0][-1]["content"]
    assert r["status"] == "abstained" and r["abstain_reason"] == "schema"
    assert r["answer"] == ABSTAIN_MESSAGES["hi-Latn"]


def test_repair_succeeds(index_dir):
    r = AnswerEngine(FakeGen(["oops", GOOD]), Retriever(index_dir)).answer(Q, "A2").response
    assert r["status"] == "ok"


def test_retrieval_gate_skips_generation(index_dir):
    gen = FakeGen([])
    r = AnswerEngine(gen, Retriever(index_dir), tau=1e9).answer(Q, "A3").response
    assert r["status"] == "abstained" and r["abstain_reason"] == "retrieval" and not gen.calls


def test_timeout_is_per_arm_status(index_dir):
    r = AnswerEngine(FakeGen([TimeoutGeneration("A0")]), Retriever(index_dir)).answer(Q, "A0").response
    assert r["status"] == "timeout"


def test_given_passages_bypass_retrieval():
    passages = [{"pid": "rbi-11693-p15", "text": "UPI: auto reversal latest on T + 1 day."}]
    r = AnswerEngine(FakeGen([GOOD]), retriever=None).answer(Q, "A2", passages=passages).response
    assert r["status"] == "ok"


TAMIL_Q = "UPI பணம் எப்போது திரும்ப வரும்?"
UPI_PASSAGES = [{"pid": "rbi-11693-p15", "text": "UPI: auto reversal latest on T + 1 day."}]


def _tamil_answer(answer):
    out = json.dumps({"answerable": True, "answer": answer, "cited_pids": ["rbi-11693-p15"], "lang": "en"})
    return AnswerEngine(FakeGen([out]), retriever=None).answer(TAMIL_Q, "A3", passages=UPI_PASSAGES).response


def test_other_language_english_answer_served_with_note():
    r = _tamil_answer("Auto reversal happens latest on T + 1 day [rbi-11693-p15]")
    assert r["status"] == "ok" and "Hinglish" in r["note"]


def test_other_language_non_english_answer_abstains_in_english():
    r = _tamil_answer("T + 1 நாளில் பணம் திரும்ப வரும் [rbi-11693-p15]")
    assert r["status"] == "abstained" and r["abstain_reason"] == "language"
    assert r["answer"] == ABSTAIN_MESSAGES["en"] and r["lang"] == "en" and "Hinglish" in r["note"]


def test_other_language_numeric_answer_served():
    r = _tamil_answer("10,000")
    assert r["status"] == "ok" and "note" in r


def test_other_language_note_on_gate_and_timeout(index_dir):
    gated = AnswerEngine(FakeGen([]), Retriever(index_dir), tau=1e9).answer(TAMIL_Q, "A3").response
    timed = AnswerEngine(FakeGen([TimeoutGeneration("A3")]), retriever=None).answer(
        TAMIL_Q, "A3", passages=UPI_PASSAGES).response
    assert gated["abstain_reason"] == "retrieval" and "Hinglish" in gated["note"]
    assert gated["answer"] == ABSTAIN_MESSAGES["en"]
    assert timed["status"] == "timeout" and "Hinglish" in timed["note"]
