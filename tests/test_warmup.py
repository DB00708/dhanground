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


def server(clock, mode="gpu", health_fails=0, arm_status=None, source="live"):
    state = {"health_calls": 0, "asks": []}

    def handler(req: httpx.Request):
        if req.url.path == "/v1/health":
            state["health_calls"] += 1
            if state["health_calls"] <= health_fails:
                raise httpx.ConnectError("refused", request=req)
            return httpx.Response(200, json={"mode": mode, "model": "m", "adapters": {}, "index_version": "v1",
                                             "cached_questions": 30})
        if req.url.path == "/v1/examples":
            return httpx.Response(200, json={"examples": EXAMPLES})
        if req.url.path == "/v1/ask":
            body = json.loads(req.content)
            state["asks"].append((clock(), req.headers["x-session-id"], body))
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
