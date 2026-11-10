"""Component 8: answer service. FE <> BE contract, /v1.

Env: INDEX_DIR, BASE_MODEL, ADAPTER_A1..A3 (see serving/loader.py), DEMO_MODE=gpu|cached,
DEMO_CACHE (default data/demo_cache.json).
Run: uvicorn apps.api.main:app --port 8000
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.pii import for_log, mask
from serving.engine import ARMS

MAX_CHARS = 500
ASK_PER_MIN = 10
COMPARE_PER_MIN = 3
DEDUPE_S = 60

log = logging.getLogger("dhanground")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class ApiError(Exception):
    def __init__(self, code: str, http: int, message: str, retry_after_s: int | None = None):
        self.code, self.http, self.message, self.retry_after_s = code, http, message, retry_after_s


class AskRequest(BaseModel):
    request_id: str
    question: str
    arm: Literal["A0", "A1", "A2", "A3"] = "A3"
    show_passages: bool = False


class CompareRequest(BaseModel):
    request_id: str
    question: str
    show_passages: bool = False


class Service:
    def __init__(self):
        self.engine = None
        self.mode = os.environ.get("DEMO_MODE", "gpu")
        self.lock = threading.Lock()
        self.recent: dict[str, tuple[float, dict]] = {}
        self.hits: dict[tuple[str, str], deque] = defaultdict(deque)
        cache_path = Path(os.environ.get("DEMO_CACHE", "data/demo_cache.json"))
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        ex = Path("configs/demo_examples.yaml")
        self.examples = yaml.safe_load(ex.read_text(encoding="utf-8"))["examples"] if ex.exists() else []
        if self.mode == "gpu":
            try:
                from serving.loader import build_engine

                self.engine = build_engine()
            except Exception as e:  # GPU or weights unavailable: serve labeled cached answers
                log.warning(json.dumps({"event": "engine_unavailable", "error": f"{type(e).__name__}: {e}"}))
                self.mode = "cached"

    def rate_limit(self, session: str, kind: str, limit: int) -> None:
        now = time.monotonic()
        q = self.hits[(session, kind)]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            wait = int(60 - (now - q[0])) + 1
            raise ApiError("RATE_LIMITED", 429, f"Try again in {wait} s", wait)
        q.append(now)

    def check_question(self, question: str) -> str:
        q = question.strip()
        if not q or len(q) > MAX_CHARS:
            raise ApiError("BAD_INPUT", 400, f"question must be 1 to {MAX_CHARS} characters")
        if not any(ch.isalpha() for ch in q):
            raise ApiError("BAD_INPUT", 400, "unsupported script")
        return mask(q)[0]

    def dedupe(self, request_id: str) -> dict | None:
        now = time.monotonic()
        for rid in [r for r, (t, _) in self.recent.items() if now - t > DEDUPE_S]:
            del self.recent[rid]
        hit = self.recent.get(request_id)
        return hit[1] if hit else None

    def cached(self, question: str, arm: str, request_id: str) -> dict:
        entry = self.cache.get(normalize(question), {}).get(arm)
        if not entry:
            raise ApiError("MODEL_UNAVAILABLE", 503, "Model is offline; try one of the example questions")
        return {**entry, "request_id": request_id, "arm": arm, "source": "cached"}

    def answer(self, question: str, arm: str, request_id: str, show_passages: bool) -> dict:
        if self.engine is None:
            return self.cached(question, arm, request_id)
        with self.lock:
            return self.engine.answer(question, arm, request_id=request_id, show_passages=show_passages).response


def normalize(question: str) -> str:
    return " ".join(question.lower().split())


def session_of(request: Request) -> str:
    return request.headers.get("x-session-id") or (request.client.host if request.client else "anon")


def log_line(resp: dict, question: str) -> None:
    log.info(json.dumps({
        "request_id": resp.get("request_id"), "arm": resp.get("arm"), "latency_ms": resp.get("latency_ms"),
        "status": resp.get("status"), "checks": resp.get("checks"), "source": resp.get("source"),
        "retrieval_top_score": resp.get("retrieval_top_score"), "question": for_log(question),
    }, ensure_ascii=False))


svc: Service | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global svc
    svc = Service()
    yield


app = FastAPI(title="DhanGround", version="1", lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError):
    body = {"request_id": getattr(request.state, "request_id", None), "status": "error",
            "error": {"code": exc.code, "message": exc.message}}
    if exc.retry_after_s is not None:
        body["error"]["retry_after_s"] = exc.retry_after_s
    return JSONResponse(body, status_code=exc.http)


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    log.exception(json.dumps({"request_id": rid, "event": "internal_error"}))
    return JSONResponse({"request_id": rid, "status": "error",
                         "error": {"code": "INTERNAL", "message": "Internal error"}}, status_code=500)


@app.post("/v1/ask")
async def ask(body: AskRequest, request: Request):
    request.state.request_id = body.request_id
    if (prev := svc.dedupe(body.request_id)) is not None:
        return prev
    svc.rate_limit(session_of(request), "ask", ASK_PER_MIN)
    question = svc.check_question(body.question)
    resp = await run_in_threadpool(svc.answer, question, body.arm, body.request_id, body.show_passages)
    svc.recent[body.request_id] = (time.monotonic(), resp)
    log_line(resp, question)
    return resp


@app.post("/v1/compare")
async def compare(body: CompareRequest, request: Request):
    request.state.request_id = body.request_id
    if (prev := svc.dedupe(body.request_id)) is not None:
        return prev
    svc.rate_limit(session_of(request), "compare", COMPARE_PER_MIN)
    question = svc.check_question(body.question)
    results = []
    for arm in ARMS:
        try:
            r = await run_in_threadpool(svc.answer, question, arm, body.request_id, body.show_passages)
        except ApiError as e:
            r = {"request_id": body.request_id, "arm": arm, "status": "error",
                 "error": {"code": e.code, "message": e.message}}
        results.append(r)
        log_line(r, question)
    resp = {"request_id": body.request_id, "results": results}
    svc.recent[body.request_id] = (time.monotonic(), resp)
    return resp


@app.get("/v1/examples")
async def examples():
    return {"examples": svc.examples}


@app.get("/v1/health")
async def health():
    from serving.loader import adapter_paths

    adapters = {arm: hashlib.sha256(str(p).encode()).hexdigest()[:12] for arm, p in adapter_paths().items()}
    engine = svc.engine
    return {
        "mode": "gpu" if engine else "cached",
        "model": os.environ.get("BASE_MODEL") or yaml.safe_load(Path("configs/arms.yaml").read_text())["base_model"],
        "adapters": adapters,
        "index_version": getattr(getattr(engine, "retriever", None), "index_version", None),
        "cached_questions": len(svc.cache),
    }
