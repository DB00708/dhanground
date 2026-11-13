"""Answer path shared by the API, the evaluator and the pair miner (BE <> AI contract).

RECEIVED -> VALIDATED -> RETRIEVED -> (GATED_ABSTAIN | GENERATED) -> CHECKED -> RETURNED
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from core import lang as langmod
from core.schema import ABSTAIN_MESSAGES, parse_output, validate
from core.template import Passage, render_user

ARMS = ("A0", "A1", "A2", "A3")
MAX_NEW_TOKENS = 384
ARM_TIMEOUT_S = 10.0
UNSUPPORTED_NOTE = "Supported languages: Hindi, Hinglish and English. Answering in English."

REPAIR_SUFFIX = "\n\nYour previous reply was not valid: {error}\nReturn one JSON object and nothing else."


class Generator(Protocol):
    def generate(self, messages: list[dict], arm: str, max_time: float) -> str: ...


class TimeoutGeneration(Exception):
    pass


class HFGenerator:
    """One base model with LoRA adapters hot-swapped per request; A0 runs with adapters disabled."""

    def __init__(self, base_model: str, adapters: dict[str, str], device: str = "cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map=device)
        self.arms = {"A0"}
        if adapters:
            first, *rest = adapters.items()
            model = PeftModel.from_pretrained(model, first[1], adapter_name=first[0])
            for name, path in rest:
                model.load_adapter(path, adapter_name=name)
            self.arms |= set(adapters)
        self.model = model.eval()

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages: list[dict], arm: str, max_time: float = ARM_TIMEOUT_S) -> str:
        if arm not in self.arms:
            raise ValueError(f"arm {arm} not loaded")
        inputs = self.tokenizer(self.render(messages), return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, max_time=max_time,
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        start = time.monotonic()
        with self.torch.inference_mode():
            if arm == "A0" and hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    out = self.model.generate(**inputs, **kwargs)
            else:
                if hasattr(self.model, "set_adapter"):
                    self.model.set_adapter(arm)
                out = self.model.generate(**inputs, **kwargs)
        if time.monotonic() - start >= max_time:
            raise TimeoutGeneration(arm)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@dataclass
class EngineResult:
    response: dict
    raw_outputs: list[str]


class AnswerEngine:
    def __init__(self, generator: Generator, retriever=None, tau: float | None = None, k: int = 4):
        self.generator, self.retriever, self.tau, self.k = generator, retriever, tau, k

    def answer(self, question: str, arm: str, request_id: str = "", passages: list[dict] | None = None,
               show_passages: bool = False) -> EngineResult:
        q_lang = langmod.detect(question)
        result = self._answer(question, q_lang, arm, request_id, passages, show_passages)
        if q_lang == langmod.OTHER:
            result.response["note"] = UNSUPPORTED_NOTE
        return result

    def _answer(self, question: str, q_lang: str, arm: str, request_id: str, passages: list[dict] | None,
                show_passages: bool) -> EngineResult:
        start = time.monotonic()
        reply_lang = langmod.reply_lang(q_lang)
        base = {"request_id": request_id, "arm": arm, "source": "live",
                "index_version": getattr(self.retriever, "index_version", None)}
        meta: dict[str, dict] = {}

        if passages is None:
            hits = self.retriever.search(question, k=self.k)
            passages = [{"pid": h.pid, "text": h.text, "score": h.score} for h in hits]
            meta = {h.pid: {"title": h.title, "url": h.url} for h in hits}
            base["retrieval_top_score"] = round(self.retriever.top_gate_score(hits), 4)
            if not hits or (self.tau is not None and self.retriever.top_gate_score(hits) < self.tau):
                return EngineResult(self._abstained(base, reply_lang, "retrieval", start, passages, show_passages, {
                    "schema": "n/a", "citations": "n/a", "claim_check": "n/a"}), [])

        typed = [Passage(p["pid"], p["text"]) for p in passages]
        messages = [{"role": "user", "content": render_user(question, typed)}]
        raw_outputs = []
        try:
            raw = self.generator.generate(messages, arm, ARM_TIMEOUT_S)
            raw_outputs.append(raw)
            output, error = parse_output(raw)
            if output is None:
                remaining = max(1.0, ARM_TIMEOUT_S - (time.monotonic() - start))
                repair = messages + [{"role": "assistant", "content": raw},
                                     {"role": "user", "content": REPAIR_SUFFIX.format(error=error)}]
                raw = self.generator.generate(repair, arm, remaining)
                raw_outputs.append(raw)
                output, error = parse_output(raw)
        except TimeoutGeneration:
            return EngineResult({**base, "status": "timeout", "answerable": None, "answer": "", "lang": reply_lang,
                                 "citations": [], "checks": {}, "latency_ms": _ms(start)}, raw_outputs)

        if output is None:
            return EngineResult(self._abstained(base, reply_lang, "schema", start, passages, show_passages, {
                "schema": "fail", "citations": "n/a", "claim_check": "n/a"}), raw_outputs)

        v = validate(output, typed, question, question_lang=q_lang)
        if q_lang == langmod.OTHER and v.status == "ok" and not langmod.answer_lang_ok(v.answer, langmod.EN):
            return EngineResult(self._abstained(base, "en", "language", start, passages, show_passages, {
                k: v.checks.get(k, "n/a") for k in ("schema", "citations", "claim_check")}), raw_outputs)
        by_pid = {p["pid"]: p for p in passages}
        response = {
            **base,
            "status": v.status,
            "answerable": v.answerable,
            "answer": v.answer,
            "lang": v.lang,
            "citations": [{"pid": pid, "title": meta.get(pid, {}).get("title"), "url": meta.get(pid, {}).get("url"),
                           "snippet": by_pid[pid]["text"][:240]} for pid in v.cited_pids],
            "checks": {k: v.checks.get(k, "n/a") for k in ("schema", "citations", "claim_check")},
            "latency_ms": _ms(start),
        }
        if v.abstain_reason:
            response["abstain_reason"] = v.abstain_reason
        if v.unsupported_claims:
            response["unsupported_claims"] = v.unsupported_claims
        if "lang" in v.checks:
            response["lang_check"] = v.checks["lang"]
        if show_passages:
            response["passages"] = passages
        return EngineResult(response, raw_outputs)

    def _abstained(self, base, lang, reason, start, passages, show_passages, checks) -> dict:
        r = {**base, "status": "abstained", "answerable": False, "answer": ABSTAIN_MESSAGES[lang], "lang": lang,
             "citations": [], "checks": checks, "abstain_reason": reason, "latency_ms": _ms(start)}
        if show_passages:
            r["passages"] = passages
        return r


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
