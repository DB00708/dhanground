"""Teacher / judge LLM calls through LiteLLM (same model strings CuratorKIT uses).

Env: TEACHER_MODEL (generation), JUDGE_MODEL (judging; must not be a Cohere model), plus the
provider's own key variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
"""
from __future__ import annotations

import json
import os
import random
import re
import time

RETRY_DELAYS_S = (2, 4, 8)


def teacher_model() -> str:
    return _require("TEACHER_MODEL")


def judge_model() -> str:
    model = _require("JUDGE_MODEL")
    if "cohere" in model.lower() or "aya" in model.lower():
        raise ValueError("JUDGE_MODEL must not be a Cohere model (judge hygiene)")
    return model


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name} to a LiteLLM model string, e.g. openai/gpt-4o-mini")
    return value


def complete(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
    import litellm

    litellm.drop_params = True  # some models (e.g. Gemini 3.x) reject a custom temperature
    for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if delay is None:
                raise
            time.sleep(delay + random.uniform(0, delay / 2))
    raise AssertionError("unreachable")


_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.S)


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(1))
