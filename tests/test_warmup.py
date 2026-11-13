import json

import httpx

from scripts import warmup

EXAMPLES = [{"tag": t, "lang": "en", "question": f"{t} question {i}?"}
            for t in ("happy", "abstain", "attack") for i in range(2)]


class Clock:
    """Fake clock; sleep advances it so pacing and backoff run without real waiting."""

    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def server(clock, mode="gpu", health_fails=0, arm_status=None, source="live", examples=None, ask_hook=None):
    state = {"health_calls": 0, "asks": []}

    def handler(req: httpx.Request):
        if req.url.path == "/v1/health":
            state["health_calls"] += 1
            if state["health_calls"] <= health_fails:
                raise httpx.ConnectError("refused", request=req)
            return httpx.Response(200, json={"mode": mode, "model": "m", "adapters": {}, "index_version": "v1",
                                             "cached_questions": 30})
        if req.url.path == "/v1/examples":
            return examples or httpx.Response(200, json={"examples": EXAMPLES})
        if req.url.path == "/v1/ask":
            body = json.loads(req.content)
            state["asks"].append((clock(), req.headers["x-session-id"], body))
            if ask_hook and (resp := ask_hook(req, body, len(state["asks"]))) is not None:
                return resp
            status = (arm_status or {}).get(body["arm"], "ok")
            return httpx.Response(200, json={"request_id": body["request_id"], "arm": body["arm"], "status": status,
                                             "source": source, "latency_ms": 1200})
        return httpx.Response(404)

    return handler, state


def run(handler, clock, **kw):
    lines = []
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://space")
    code = warmup.run("http://space", client=client, sleep=clock.sleep, clock=clock, out=lines.append, **kw)
    return code, "\n".join(lines)


def test_healthy_gpu_exits_zero_and_respects_rate_limit():
    clock = Clock()
    handler, state = server(clock)
    code, out = run(handler, clock)
    assert code == 0 and "OK:" in out
    asks = state["asks"]
    assert len(asks) == 12  # 3 tags x 1 per tag x 4 arms
    assert {a[2]["arm"] for a in asks} == set(warmup.ARMS)
    assert len({a[1] for a in asks}) == 1  # one dedicated session id
    times = [a[0] for a in asks]
    assert all(sum(1 for u in times if t - 60 <= u <= t) <= 10 for t in times)  # never >10 per minute


def test_cached_mode_exits_nonzero_without_asking():
    clock = Clock()
    handler, state = server(clock, mode="cached")
    code, out = run(handler, clock)
    assert code != 0 and "GPU model did not load" in out and not state["asks"]


def test_arm_timeout_exits_nonzero():
    clock = Clock()
    handler, _ = server(clock, arm_status={"A2": "timeout"})
    code, out = run(handler, clock)
    assert code != 0 and "A2/happy: status timeout" in out


def test_cached_source_exits_nonzero():
    clock = Clock()
    handler, _ = server(clock, source="cached")
    code, out = run(handler, clock)
    assert code != 0 and "source cached" in out


def test_health_retries_until_up():
    clock = Clock()
    handler, state = server(clock, health_fails=3)
    code, out = run(handler, clock)
    assert code == 0 and state["health_calls"] == 4
    assert clock.sleeps[:3] == [2.0, 4.0, 8.0]  # exponential backoff, no real sleeping


def test_health_never_up_gives_up():
    clock = Clock()
    handler, _ = server(clock, health_fails=10_000)
    code, out = run(handler, clock, max_wait=120)
    assert code == warmup.EXIT_NOT_READY and "never answered" in out and clock.t <= 120


def rate_limited(retry_after=7):
    return httpx.Response(429, json={"request_id": None, "status": "error",
                                     "error": {"code": "RATE_LIMITED", "message": "x", "retry_after_s": retry_after}})


def test_429_sleeps_retry_after_once_then_succeeds():
    clock = Clock()
    handler, state = server(clock, ask_hook=lambda req, body, n: rate_limited() if n == 1 else None)
    code, _ = run(handler, clock)
    assert code == 0 and len(state["asks"]) == 13 and clock.sleeps[0] == 7.0
    assert state["asks"][0][2]["request_id"] == state["asks"][1][2]["request_id"]  # same request retried
    assert state["asks"][0][2]["show_passages"] is False


def test_second_429_fails():
    clock = Clock()
    handler, state = server(clock, ask_hook=lambda req, body, n: rate_limited() if n <= 2 else None)
    code, out = run(handler, clock)
    assert code == 1 and "A0/happy: status error (RATE_LIMITED)" in out
    # one retry only (2 sends for A0, then 11 more arms/tags); the retry waited retry_after_s
    assert len(state["asks"]) == 13 and state["asks"][1][0] - state["asks"][0][0] == 7.0
    assert state["asks"][2][2]["request_id"] != state["asks"][1][2]["request_id"]


def test_examples_missing_tag_fails():
    clock = Clock()
    only_happy = httpx.Response(200, json={"examples": [e for e in EXAMPLES if e["tag"] == "happy"]})
    handler, state = server(clock, examples=only_happy)
    code, out = run(handler, clock)
    assert code == 1 and "abstain, attack" in out and not state["asks"]


def test_examples_error_fails():
    for resp in (httpx.Response(500, json={"status": "error", "error": {"code": "INTERNAL", "message": "x"}}),
                 httpx.Response(200, json=[1, 2])):
        clock = Clock()
        handler, state = server(clock, examples=resp)
        code, out = run(handler, clock)
        assert code == 1 and f"FAIL: /v1/examples HTTP {resp.status_code}" in out and not state["asks"]


def test_transport_timeout_is_status_timeout():
    def hook(req, body, n):
        if body["arm"] == "A1":
            raise httpx.ReadTimeout("slow", request=req)

    clock = Clock()
    handler, _ = server(clock, ask_hook=hook)
    code, out = run(handler, clock)
    assert code == 1 and "A1/happy: status timeout (client timeout)" in out


def test_examples_non_json_or_missing_list_fails():
    for resp in (httpx.Response(200, text="<html>loading</html>"), httpx.Response(200, json={"items": []})):
        clock = Clock()
        handler, state = server(clock, examples=resp)
        code, out = run(handler, clock)
        assert code == 1 and "FAIL: /v1/examples HTTP 200" in out and not state["asks"]


def test_429_retry_is_recorded_in_pacer():
    sent = []
    clock = Clock()
    pacer = warmup.Pacer(clock.sleep, clock)
    real_wait = pacer.wait
    pacer.wait = lambda: sent.append(clock()) or real_wait()
    calls = []

    def handler(req):
        calls.append(req)
        return rate_limited() if len(calls) == 1 else httpx.Response(200, json={"status": "ok", "source": "live"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://space")
    r = warmup.ask(client, "s", "q?", "A0", clock.sleep, clock, pacer)
    assert r["status"] == "ok" and len(calls) == 2 and len(pacer.sent) == 1 and sent == [7.0]


def test_last_health_get_timeout_capped_at_remaining_wait():
    clock = Clock()
    timeouts = []

    def handler(req):
        timeouts.append(req.extensions["timeout"]["read"])
        clock.t += 50  # each GET burns 50 s
        raise httpx.ConnectError("refused", request=req)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://space", timeout=60.0)
    code = warmup.run("http://space", client=client, sleep=clock.sleep, clock=clock, out=lambda _: None, max_wait=120)
    assert code == warmup.EXIT_NOT_READY
    assert timeouts[0] == 60.0 and all(t <= 60.0 for t in timeouts) and min(timeouts) < 60.0
