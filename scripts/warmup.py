"""Demo Day warm-up (design edge case 18: GPU Space cold start). Run ~10 min before the slot.

1. Polls GET /v1/health with backoff until it answers; fails if mode is "cached" (GPU did not load).
2. Sends one POST /v1/ask per arm (A0..A3) for --per-tag example questions of each tag
   (happy, abstain, attack) from /v1/examples, paced under the 10/min ask rate limit.
Exit 0 only if health is gpu and every answer is ok|abstained with source "live".

Run: python -m scripts.warmup --url https://<space>.hf.space
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import deque
from typing import Callable

import httpx

ARMS = ("A0", "A1", "A2", "A3")
TAGS = ("happy", "abstain", "attack")
OK_STATUSES = ("ok", "abstained")
ASK_BUDGET = 9  # per window; the server allows 10/min per session, keep one spare
WINDOW_S = 61.0
HEALTH_MAX_WAIT_S = 900.0  # cold start (image pull + weights) can take minutes
BACKOFF_START_S, BACKOFF_CAP_S = 2.0, 30.0

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def wait_for_health(client: httpx.Client, sleep: Callable, clock: Callable, out: Callable,
                    max_wait: float = HEALTH_MAX_WAIT_S) -> dict | None:
    start, delay, attempt = clock(), BACKOFF_START_S, 0
    while True:
        attempt += 1
        remaining = max(max_wait - (clock() - start), 0.1)
        configured = client.timeout.read
        try:  # cap each GET so the total wait stays within max_wait
            r = client.get("/v1/health", timeout=min(configured, remaining) if configured else remaining)
            if r.status_code == 200:
                return r.json()
            why = f"HTTP {r.status_code}"
        except (httpx.HTTPError, ValueError) as e:  # connection refused, timeout, non-JSON page
            why = type(e).__name__
        waited = clock() - start
        if waited + delay > max_wait:
            out(f"health: not up after {waited:.0f} s ({attempt} attempts, last: {why})")
            return None
        out(f"health: attempt {attempt} not up ({why}); retry in {delay:.0f} s (waited {waited:.0f} s)")
        sleep(delay)
        delay = min(delay * 2, BACKOFF_CAP_S)


def pick_questions(examples: list[dict], per_tag: int) -> list[dict]:
    picked = []
    for tag in TAGS:
        picked += [e for e in examples if e.get("tag") == tag][:per_tag]
    return picked


class Pacer:
    """Client-side sliding window so the warm-up never hits the ask rate limit."""

    def __init__(self, sleep: Callable, clock: Callable, budget: int = ASK_BUDGET, window: float = WINDOW_S):
        self.sleep, self.clock, self.budget, self.window = sleep, clock, budget, window
        self.sent: deque[float] = deque()

    def wait(self) -> None:
        while self.sent and self.clock() - self.sent[0] >= self.window:
            self.sent.popleft()
        if len(self.sent) >= self.budget:
            self.sleep(self.window - (self.clock() - self.sent[0]))
            self.sent.popleft()
        self.sent.append(self.clock())


def ask(client: httpx.Client, session: str, question: str, arm: str, sleep: Callable, clock: Callable,
        pacer: Pacer) -> dict:
    body = {"request_id": f"warmup-{uuid.uuid4().hex}", "question": question, "arm": arm, "show_passages": False}
    t0 = clock()
    for attempt in range(2):
        try:
            r = client.post("/v1/ask", json=body, headers={"x-session-id": session})
        except httpx.TimeoutException:
            return {"status": "timeout", "source": "-", "latency_ms": round((clock() - t0) * 1000), "reason": "client timeout"}
        except httpx.HTTPError as e:
            return {"status": "error", "source": "-", "latency_ms": None, "reason": type(e).__name__}
        try:
            data = r.json()
        except ValueError:
            return {"status": "error", "source": "-", "latency_ms": None, "reason": f"HTTP {r.status_code} non-JSON"}
        err = data.get("error") or {}
        if r.status_code == 429 and attempt == 0:  # pacing should prevent this; honour retry_after once
            sleep(float(err.get("retry_after_s", 60)))
            pacer.wait()  # the retry is a send too
            continue
        if data.get("status") == "error":
            data["reason"] = err.get("code", f"HTTP {r.status_code}")
        return data
    return data


def run(url: str, per_tag: int = 1, timeout: float = 60.0, client: httpx.Client | None = None,
        sleep: Callable = time.sleep, clock: Callable = time.monotonic, out: Callable = print,
        max_wait: float = HEALTH_MAX_WAIT_S) -> int:
    own = client is None
    client = client or httpx.Client(base_url=url, timeout=timeout)
    try:
        return _run(client, per_tag, sleep, clock, out, max_wait)
    finally:
        if own:
            client.close()


def _run(client, per_tag, sleep, clock, out, max_wait) -> int:
    health = wait_for_health(client, sleep, clock, out, max_wait)
    if health is None:
        out("FAIL: /v1/health never answered; the Space is not up")
        return EXIT_NOT_READY
    out(f"health: mode={health.get('mode')} model={health.get('model')} index={health.get('index_version')}")
    if health.get("mode") != "gpu":
        out(f"FAIL: mode is {health.get('mode')!r}, not 'gpu'; the GPU model did not load (UI would serve cached answers)")
        return EXIT_NOT_READY

    try:
        r = client.get("/v1/examples")
    except httpx.HTTPError as e:
        out(f"FAIL: could not reach /v1/examples ({type(e).__name__})")
        return EXIT_FAILED
    try:
        body = r.json() if r.status_code == 200 else None
    except ValueError:
        body = None
    if not isinstance(body, dict) or not isinstance(body.get("examples"), list):
        out(f"FAIL: /v1/examples HTTP {r.status_code}" + ("" if r.status_code != 200 else " (unexpected body)"))
        return EXIT_FAILED
    examples = [e for e in body["examples"] if isinstance(e, dict)]
    questions = pick_questions(examples, per_tag)
    missing = [t for t in TAGS if not any(q["tag"] == t for q in questions)]
    if missing:
        out(f"FAIL: /v1/examples has no questions for tag(s) {', '.join(missing)}")
        return EXIT_FAILED

    session, pacer, rows = f"warmup-{uuid.uuid4().hex[:8]}", Pacer(sleep, clock), []
    out(f"asking {len(questions)} question(s) x {len(ARMS)} arms as session {session} (<= {ASK_BUDGET}/min)")
    for q in questions:
        for arm in ARMS:
            pacer.wait()
            resp = ask(client, session, q["question"], arm, sleep, clock, pacer)
            rows.append((arm, q["tag"], resp))

    failures = []
    out(f"{'arm':<4} {'tag':<8} {'status':<10} {'source':<7} {'latency_ms':>10}  note")
    for arm, tag, r in rows:
        status, source = r.get("status", "?"), r.get("source", "?")
        bad = []
        if status not in OK_STATUSES:
            bad.append(f"status {status}" + (f" ({r['reason']})" if r.get("reason") else ""))
        if source != "live":
            bad.append(f"source {source}")
        if bad:
            failures.append(f"{arm}/{tag}: {', '.join(bad)}")
        lat = r.get("latency_ms")
        out(f"{arm:<4} {tag:<8} {status:<10} {source:<7} {lat if lat is not None else '-':>10}  {'; '.join(bad)}")
    for arm in ARMS:
        lats = [r.get("latency_ms") for a, _, r in rows if a == arm and isinstance(r.get("latency_ms"), (int, float))]
        if lats:
            out(f"{arm}: max latency {max(lats)} ms over {len(lats)} answer(s)")

    if failures:
        out(f"FAIL: {len(failures)} of {len(rows)} request(s) not ok/abstained+live: " + "; ".join(failures))
        return EXIT_FAILED
    out(f"OK: GPU warm, {len(rows)} live answers across {len(ARMS)} arms")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Warm up the DhanGround Space before a demo slot.")
    p.add_argument("--url", default="http://127.0.0.1:7860", help="base URL of the API/Space")
    p.add_argument("--per-tag", type=int, default=1, help="example questions per tag (happy, abstain, attack)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-request HTTP timeout in seconds")
    a = p.parse_args(argv)
    return run(a.url.rstrip("/"), per_tag=a.per_tag, timeout=a.timeout)


if __name__ == "__main__":
    sys.exit(main())
