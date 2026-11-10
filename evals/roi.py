"""ROI: self-hosted Tiny Aya + LoRA vs a closed model, per 1M answers.

  Self-host = 1e6 / answers_per_hour * gpu_price_per_hour
  Closed    = 1e6 * (in_tokens * P_in + out_tokens * P_out) / 1e6

Throughput is measured here, not assumed. Closed-model prices are arguments: pass the vendor's
published rate on the day and cite the page in the write-up.

Usage: python -m evals.roi --measure --arm A3 --n 100 --p-in 0.15 --p-out 0.60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def self_host_cost(answers_per_hour: float, gpu_price_per_hour: float) -> float:
    return 1e6 / answers_per_hour * gpu_price_per_hour


def closed_cost(in_tokens: float, out_tokens: float, p_in_per_m: float, p_out_per_m: float) -> float:
    return 1e6 * (in_tokens * p_in_per_m + out_tokens * p_out_per_m) / 1e6


def measure(arm: str, n: int) -> dict:
    from serving.loader import build_engine

    engine = build_engine(arms=[arm] if arm != "A0" else [])
    rows = [json.loads(l) for l in Path("data/arms/dev.jsonl").open(encoding="utf-8")][:n]
    tok = engine.generator.tokenizer
    in_tok = out_tok = 0
    lat = []
    start = time.monotonic()
    for r in rows:
        t = time.monotonic()
        res = engine.answer(r["question"], arm, passages=r["passages"])
        lat.append(time.monotonic() - t)
        from core.template import Passage, render_user

        in_tok += len(tok(render_user(r["question"], [Passage(p["pid"], p["text"]) for p in r["passages"]]))["input_ids"])
        out_tok += sum(len(tok(o)["input_ids"]) for o in res.raw_outputs)
    elapsed = time.monotonic() - start
    lat.sort()
    return {"arm": arm, "n": len(rows), "answers_per_hour_sequential": 3600 * len(rows) / elapsed,
            "p50_s": lat[len(lat) // 2], "p95_s": lat[int(0.95 * (len(lat) - 1))],
            "avg_in_tokens": in_tok / len(rows), "avg_out_tokens": out_tok / len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--arm", default="A3")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--answers-per-hour", type=float, help="use a measured batched-serving number instead")
    ap.add_argument("--gpu-price", type=float, default=0.80, help="USD per hour (HF L4)")
    ap.add_argument("--p-in", type=float, required=True, help="closed model USD per 1M input tokens")
    ap.add_argument("--p-out", type=float, required=True, help="closed model USD per 1M output tokens")
    ap.add_argument("--in-tokens", type=float, default=2500)
    ap.add_argument("--out-tokens", type=float, default=150)
    a = ap.parse_args()

    m = measure(a.arm, a.n) if a.measure else {}
    aph = a.answers_per_hour or m.get("answers_per_hour_sequential")
    if not aph:
        raise SystemExit("pass --measure or --answers-per-hour")
    in_t, out_t = m.get("avg_in_tokens", a.in_tokens), m.get("avg_out_tokens", a.out_tokens)
    res = {**m, "answers_per_hour": aph, "self_host_per_1m": self_host_cost(aph, a.gpu_price),
           "closed_per_1m": closed_cost(in_t, out_t, a.p_in, a.p_out),
           "inputs": {"gpu_price": a.gpu_price, "p_in": a.p_in, "p_out": a.p_out, "in_tokens": in_t, "out_tokens": out_t}}
    res["ratio_closed_over_self"] = res["closed_per_1m"] / res["self_host_per_1m"]
    out = Path("evals/results/roi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
