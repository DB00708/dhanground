"""Pool seeds per arm, report mean with a 95% bootstrap CI, check the regression gates, draw the headline chart.

Usage: python -m evals.summarize --split test
Writes evals/results/summary_{split}.json, summary_{split}.md and headline_{split}.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals import metrics

RESULTS = Path("evals/results")
ARMS = ("A0", "A1", "A2", "A3")


def load_arm(arm: str, split: str) -> tuple[list[dict], list[int]]:
    rows, seeds = [], []
    for f in sorted(RESULTS.glob(f"{arm}_s*_{split}_verdicts.jsonl")):
        seeds.append(int(f.name.split("_s")[1].split("_")[0]))
        rows.extend(json.loads(l) for l in f.open(encoding="utf-8"))
    return rows, seeds


def seed_backends(arm: str, seeds: list[int], split: str) -> dict[str, str]:
    """{seed: backend} from each seed's result JSON (run_eval records it from the adapter's dhanground.json)."""
    out = {}
    for s in seeds:
        f = RESULTS / f"{arm}_s{s}_{split}.json"
        out[str(s)] = json.loads(f.read_text(encoding="utf-8")).get("backend", "unknown") if f.exists() else "unknown"
    return out


def regression_gates(prev: dict | None, cur: dict, general_drop: float | None) -> dict:
    m = {k: v["value"] for k, v in cur.items()}
    gates = {
        "schema_valid>=0.98": m["schema_valid"] is not None and m["schema_valid"] >= 0.98,
        "over_refusal<0.10": m["over_refusal"] is not None and m["over_refusal"] < 0.10,
    }
    if prev:
        p, c = prev["unsupported_claim_rate"]["value"], m["unsupported_claim_rate"]
        gates["unsupported_claims<=previous_arm"] = p is not None and c is not None and c <= p
    if general_drop is not None:
        gates["general_capability_drop<3"] = general_drop < 3
    return gates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    general = {}
    gpath = RESULTS / "general_capability.json"
    if gpath.exists():
        general = json.loads(gpath.read_text())

    summary, prev = {}, None
    for arm in ARMS:
        rows, seeds = load_arm(arm, a.split)
        if not rows:
            continue
        rep = metrics.report(rows)
        drop = None
        if arm in general and "A0" in general:
            drop = 100 * (general["A0"] - general[arm])
        summary[arm] = {"seeds": seeds, "backends": seed_backends(arm, seeds, a.split), "n_rows": len(rows), "metrics": rep,
                        "gates": regression_gates(prev, rep, drop) if arm != "A0" else {},
                        "general_capability": general.get(arm)}
        prev = rep
    if "A1" in summary and "A2" in summary:
        c1 = summary["A1"]["metrics"]["unsupported_claim_rate"]["ci95"]
        c2 = summary["A2"]["metrics"]["unsupported_claim_rate"]["ci95"]
        summary["curation_dividend"] = {
            "unsupported_claim_rate_A1_minus_A2": (summary["A1"]["metrics"]["unsupported_claim_rate"]["value"] or 0)
            - (summary["A2"]["metrics"]["unsupported_claim_rate"]["value"] or 0),
            "real_difference": metrics.cis_overlap(c1, c2) is False,
        }
    (RESULTS / f"summary_{a.split}.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / f"summary_{a.split}.md").write_text(to_markdown(summary), encoding="utf-8")
    draw_headline(summary, RESULTS / f"headline_{a.split}.png")
    print(to_markdown(summary))


def fmt(m: dict) -> str:
    if m["value"] is None:
        return "n/a"
    ci = f" [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]" if m["ci95"] else ""
    return f"{m['value']:.3f}{ci}"


def to_markdown(summary: dict) -> str:
    arms = [a for a in ARMS if a in summary]
    names = list(metrics.METRICS)
    lines = ["| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for n in names:
        arrow = " (lower better)" if n in metrics.LOWER_IS_BETTER else ""
        lines.append(f"| {n}{arrow} | " + " | ".join(fmt(summary[a]["metrics"][n]) for a in arms) + " |")
    if any(summary[a].get("general_capability") is not None for a in arms):
        lines.append("| general_capability | " + " | ".join(
            f"{summary[a]['general_capability']:.3f}" if summary[a].get("general_capability") is not None else "n/a"
            for a in arms) + " |")
    lines.append("| backend by seed | " + " | ".join(
        ", ".join(f"s{s}: {b}" for s, b in summary[a].get("backends", {}).items()) or "n/a" for a in arms) + " |")
    return "\n".join(lines) + "\n"


def draw_headline(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ARMS if a in summary and summary[a]["metrics"]["unsupported_claim_rate"]["value"] is not None]
    if not arms:
        return
    vals = [summary[a]["metrics"]["unsupported_claim_rate"]["value"] * 100 for a in arms]
    cis = [summary[a]["metrics"]["unsupported_claim_rate"]["ci95"] for a in arms]
    err = [[v - c[0] * 100 for v, c in zip(vals, cis)], [c[1] * 100 - v for v, c in zip(vals, cis)]]
    labels = {"A0": "A0 base", "A1": "A1 raw SFT", "A2": "A2 curated SFT", "A3": "A3 curated + DPO"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.bar([labels[a] for a in arms], vals, yerr=err, capsize=4,
           color=["#9aa0a6" if a != "A2" else "#1a73e8" for a in arms])
    ax.set_ylabel("Unsupported-claim rate (%)")
    ax.set_title("Answers with a number not in the cited RBI passage")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
