import json

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from retrieval.search import Retriever
from serving.engine import AnswerEngine, TimeoutGeneration

GOOD = json.dumps({"answerable": True, "answer": "T + 1 din mein wapas [rbi-11693-p15]",
                   "cited_pids": ["rbi-11693-p15"], "lang": "hi-Latn"})


class Gen:
    def generate(self, messages, arm, max_time):
        if arm == "A0":
            raise TimeoutGeneration(arm)
        return GOOD


@pytest.fixture
def client(index_dir, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "cached")
    with TestClient(main.app) as c:
        main.svc.engine = AnswerEngine(Gen(), Retriever(index_dir))
        yield c


def ask(c, rid="r1", q="UPI payment fail hua, paisa kab wapas aayega?", **kw):
    return c.post("/v1/ask", json={"request_id": rid, "question": q, "arm": "A3", **kw},
                  headers={"x-session-id": kw.pop("session", "s1")})


def test_ask_ok(client):
    r = ask(client)
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["checks"]["claim_check"] == "pass"


def test_bad_input_shapes(client):
    for q in ["", "x" * 501, "1234 !!!"]:
        r = ask(client, rid=f"b{len(q)}", q=q)
        assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_INPUT"


def test_rate_limit_and_error_shape(client):
    for i in range(10):
        assert ask(client, rid=f"x{i}").status_code == 200
    r = ask(client, rid="x10")
    assert r.status_code == 429
    body = r.json()
    assert body["status"] == "error" and body["error"]["code"] == "RATE_LIMITED" and body["error"]["retry_after_s"] > 0


def test_request_id_dedupe(client):
    first = ask(client, rid="same").json()
    assert ask(client, rid="same", q="different question entirely").json() == first


def test_compare_isolates_arm_timeout(client):
    r = client.post("/v1/compare", json={"request_id": "c1", "question": "UPI refund kab aayega?"}).json()
    statuses = {x["arm"]: x["status"] for x in r["results"]}
    assert statuses == {"A0": "timeout", "A1": "ok", "A2": "ok", "A3": "ok"}


def test_pii_masked_before_model(client, monkeypatch):
    seen = {}
    orig = main.svc.engine.answer

    def spy(question, *a, **k):
        seen["q"] = question
        return orig(question, *a, **k)

    monkeypatch.setattr(main.svc.engine, "answer", spy)
    ask(client, rid="p1", q="mera account 123456789012 hai, UPI refund kab aayega?")
    assert "123456789012" not in seen["q"]


def test_cached_mode_without_engine(client):
    main.svc.engine = None
    main.svc.cache = {"upi refund kab aayega?": {"A3": {"status": "ok", "answer": "cached", "checks": {}}}}
    r = ask(client, rid="k1", q="UPI refund kab aayega?").json()
    assert r["source"] == "cached" and r["answer"] == "cached"
    r = ask(client, rid="k2", q="something not cached")
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_health(client):
    h = client.get("/v1/health").json()
    assert h["mode"] == "gpu" and "adapters" in h
