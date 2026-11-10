"""LLM-as-judge prompts. The judge sees passages and the gold answer, never the arm name."""
from __future__ import annotations

from typing import Callable

from intelligence import llm

UNANSWERABLE_CHECK = """Do these passages contain the answer to the question? Judge only from the passages.

Question: {question}

Passages:
{passages}

Return JSON only: {{"answered": true or false, "reason": "one short sentence"}}"""

GROUNDED_ACCURACY = """You grade an Indian banking assistant's answer.

Question: {question}

Passages the assistant was given:
{passages}

Reference answer: {gold}

Assistant answer: {prediction}

An answer is CORRECT when it gives the same rule as the reference for the same payment system or product
(a correct rule for the wrong payment system is incorrect). It is SUPPORTED when every claim in it is
stated in the passages. Language or wording differences do not matter.

Return JSON only: {{"correct": true or false, "supported": true or false, "reason": "one short sentence"}}"""


def format_passages(passages: list[dict]) -> str:
    return "\n".join(f"[{p['pid']}] {p['text']}" for p in passages)


def _call(prompt: str, complete: Callable[[str], str] | None) -> dict:
    if complete is None:
        model = llm.judge_model()
        complete = lambda p: llm.complete(model, p, temperature=0.0, max_tokens=300)  # noqa: E731
    data = llm.parse_json(complete(prompt))
    if not isinstance(data, dict):
        raise ValueError(f"judge returned {type(data).__name__}, expected an object")
    return data


def passages_answer(question: str, passages: list[dict], complete=None) -> bool:
    data = _call(UNANSWERABLE_CHECK.format(question=question, passages=format_passages(passages)), complete)
    return bool(data["answered"])


def grade(question: str, passages: list[dict], gold: str, prediction: str, complete=None) -> dict:
    data = _call(GROUNDED_ACCURACY.format(question=question, passages=format_passages(passages),
                                          gold=gold, prediction=prediction), complete)
    return {"correct": bool(data["correct"]), "supported": bool(data["supported"]),
            "reason": str(data.get("reason", ""))}
