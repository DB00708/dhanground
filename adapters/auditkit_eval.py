"""General-capability check through AuditKIT: a held-out multiple-choice benchmark, base vs each adapter.

The model is passed to AuditKIT as a list[str] -> list[str] callable that runs our own
adapter-switching generator, so every arm is scored by the same harness.

Usage: python -m adapters.auditkit_eval --arms A0 A1 A2 A3 [--dataset openai/MMMLU --config HI_IN --n 300]
Writes evals/results/general_capability.json ({arm: accuracy}).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

LETTER_RE = re.compile(r"\b([ABCD])\b")


def load_mcq(dataset: str, config: str | None, split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idx:
        r = ds[i]
        prompt = (f"{r['Question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n"
                  "Answer with the letter only.")
        out.append({"input": prompt, "target": r["Answer"].strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--dataset", default="openai/MMMLU")
    ap.add_argument("--config", default="HI_IN")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    import auditkit as ak
    from serving.loader import adapter_paths
    from serving.engine import HFGenerator
    from adapters.common import load_configs

    base = load_configs()[0]["base_model"]
    gen = HFGenerator(base, {k: v for k, v in adapter_paths().items() if k in a.arms})
    samples = [ak.Sample(input=s["input"], target=s["target"]) for s in load_mcq(a.dataset, a.config, a.split, a.n, 0)]

    out_path = Path("evals/results/general_capability.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for arm in a.arms:
        def model(prompts: list[str], arm=arm) -> list[str]:
            outs = []
            for p in prompts:
                text = gen.generate([{"role": "user", "content": p}], arm, max_time=30.0)
                m = LETTER_RE.search(text)
                outs.append(m.group(1) if m else text.strip()[:1])
            return outs

        run = ak.evaluate(samples, model, ["exact_match"])
        results[arm] = float(run.headline["exact_match"])
        print(arm, results[arm])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
