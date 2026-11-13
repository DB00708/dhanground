"""Set the retrieval gate tau on dev (design edge case 8).

Serving abstains without generation when the top gate score is below tau. We pick the highest tau
that gates at most --max-gated of answerable dev questions, so the gate uses only part of the 10%
over-refusal budget (the model also refuses on its own).

Usage: python -m evals.tune_tau --index index/<version> [--dev data/arms/dev.jsonl] [--max-gated 0.05] [--write]
Runs live retrieval for each dev question (as serving does), writes evals/results/tau.json and,
with --write, replaces only the tau value in configs/retrieval.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

RESULTS = Path("evals/results")
CONFIG = Path("configs/retrieval.yaml")
CURVE_QUANTILES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
_TAU_LINE = re.compile(r"^(tau:[ \t]*)([^\s#]*)(.*)$", re.MULTILINE)  # value may be empty
# The question at the cut sits exactly on tau; a float drift (e.g. CPU vs GPU cosine) would gate it.
# Lower tau by a tiny margin so it stays ungated; this never gates more questions on dev.
_EDGE = 1e-6


def _tau_at(answerable_sorted: list[float], max_gated: float) -> float:
    # A question is gated when score < tau. With tau = the (m+1)-th smallest score, at most m fall
    # below it (fewer with ties); any higher tau would gate m+1.
    m = math.floor(max_gated * len(answerable_sorted) + 1e-9)
    cut = answerable_sorted[min(m, len(answerable_sorted) - 1)]
    return cut - _EDGE * max(1.0, abs(cut))


def _share_below(scores: list[float], tau: float) -> float | None:
    return sum(s < tau for s in scores) / len(scores) if scores else None


def choose_tau(answerable: list[float], unanswerable: list[float], max_gated: float = 0.05) -> dict:
    """Highest tau (less a 1e-6 edge margin) gating at most max_gated of answerable scores, plus a small curve for the report."""
    if not answerable:
        raise ValueError("need at least one answerable dev question to set tau")
    if not 0.0 <= max_gated < 1.0:
        raise ValueError(f"max_gated must be in [0, 1), got {max_gated}")
    if not all(math.isfinite(s) for s in [*answerable, *unanswerable]):
        raise ValueError("gate scores must be finite")
    a = sorted(answerable)
    tau = _tau_at(a, max_gated)
    curve = [{"quantile": q, "tau": t, "answerable_gated": _share_below(a, t),
              "unanswerable_caught": _share_below(unanswerable, t)}
             for q in CURVE_QUANTILES for t in [_tau_at(a, q)]]
    return {"tau": tau, "max_gated": max_gated, "answerable_gated": _share_below(a, tau),
            "unanswerable_caught": _share_below(unanswerable, tau), "curve": curve}


def _yaml_float(x: float) -> str:
    """Shortest exact repr that YAML 1.1 (PyYAML) reads back as a float: it needs a dot in the
    mantissa, so 1e-05 must be written 1.0e-05."""
    s = repr(float(x))
    if "e" in s and "." not in s.split("e")[0]:
        mant, exp = s.split("e")
        s = f"{mant}.0e{exp}"
    return s


def replace_tau(config_text: str, tau: float) -> str:
    """Replace only the value on the top-level `tau:` line, keeping its comment and every other line."""
    def sub(m: re.Match) -> str:
        old, new, rest = m.group(2), _yaml_float(tau), m.group(3)
        comment = rest.lstrip(" \t")
        if comment.strip():  # keep the comment in its column when the new value fits
            rest = " " * max(1, len(old) + len(rest) - len(comment) - len(new)) + comment
        head = m.group(1) if m.group(1) != "tau:" else "tau: "  # a bare `tau:` needs a space
        return head + new + rest

    new_text, n = _TAU_LINE.subn(sub, config_text, count=1)
    if n != 1:
        raise ValueError("no top-level `tau:` line found in the retrieval config")
    if yaml.safe_load(new_text).get("tau") != float(tau):
        raise ValueError("rewritten config does not read back the same tau")
    return new_text


def gate_scores(index_dir: Path, dev_rows: list[dict], cfg: dict) -> tuple[list[float], list[float], dict]:
    from retrieval.search import Retriever

    r = Retriever(index_dir, cfg["hybrid"]["bm25_weight"], cfg["hybrid"]["embedding_weight"])
    ans, unans = [], []
    for row in dev_rows:
        score = r.top_gate_score(r.search(row["question"], k=cfg["k"]))
        (ans if row["answerable"] else unans).append(score)
    return ans, unans, r.meta


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:5.1f}%"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--dev", default=Path("data/arms/dev.jsonl"), type=Path)
    ap.add_argument("--max-gated", default=0.05, type=float)
    ap.add_argument("--write", action="store_true", help="update tau in the retrieval config")
    ap.add_argument("--config", default=CONFIG, type=Path)
    ap.add_argument("--out", default=RESULTS / "tau.json", type=Path)
    args = ap.parse_args(argv)

    cfg_text = args.config.read_bytes().decode("utf-8")  # keep line endings
    cfg = yaml.safe_load(cfg_text)
    rows = [json.loads(l) for l in args.dev.open(encoding="utf-8") if l.strip()]
    ans, unans, meta = gate_scores(args.index, rows, cfg)
    if not meta.get("embedding_model"):
        print("WARNING: index has no embedding model; the gate score is raw BM25, which separates "
              "answerable from unanswerable poorly and depends on question length. Treat this tau as "
              "a placeholder.", file=sys.stderr)

    res = choose_tau(ans, unans, args.max_gated)
    res.update({"n_answerable": len(ans), "n_unanswerable": len(unans),
                "index_version": meta.get("index_version"), "embedding_model": meta.get("embedding_model"),
                "gate": "cosine" if meta.get("embedding_model") else "bm25_raw", "dev": str(args.dev)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"{'quantile':>8}  {'tau':>10}  {'ans gated':>9}  {'unans caught':>12}")
    for c in res["curve"]:
        print(f"{c['quantile']:>8.2f}  {c['tau']:>10.4f}  {_pct(c['answerable_gated']):>9}  "
              f"{_pct(c['unanswerable_caught']):>12}")
    print(f"tau={res['tau']:.4f} (answerable gated {_pct(res['answerable_gated'])}, unanswerable caught "
          f"{_pct(res['unanswerable_caught'])}; n={len(ans)}/{len(unans)}) -> {args.out}")
    if args.write:
        args.config.write_bytes(replace_tau(cfg_text, res["tau"]).encode("utf-8"))
        print(f"wrote tau to {args.config}")
    return res


if __name__ == "__main__":
    main()
