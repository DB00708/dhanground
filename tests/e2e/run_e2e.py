"""End-to-end integration driver: every stage on a small slice, with a fake teacher and a fake generator.

  python tests/e2e/run_e2e.py --data-root <dir with corpus/ index/ sources/pdf/> --work <scratch dir>

Stages run as real CLI subprocesses (cwd = work dir, PYTHONPATH = repo) and each one's outputs are
asserted. Exit code 0 only if every stage passes. Report: <work>/e2e_report.json.
In Docker: docker compose -f docker/docker-compose.yml run --rm e2e
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.schema import ModelOutput, TrainingRow  # noqa: E402
from core.splits import check_no_overlap  # noqa: E402

N_ARM_ROWS = 40
INDEX_VERSION = "e2e"
FAKE_GEN = "tests.e2e.fakes:FakeGenerator"
API_FIELDS = {"request_id", "arm", "status", "answerable", "answer", "lang", "citations", "checks",
              "latency_ms", "source", "index_version"}


class StageFailed(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise StageFailed(msg)


def jsonl(path: Path) -> list[dict]:
    check(path.exists(), f"missing {path}")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


class Harness:
    def __init__(self, data_root: Path, work: Path):
        self.data_root, self.work = data_root, work
        self.env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        for k in ("TEACHER_MODEL", "JUDGE_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            self.env.pop(k, None)
        self.results: list[dict] = []
        self.splits: dict[str, list[str]] = {}

    # -- plumbing ---------------------------------------------------------------------------------
    def run(self, *args: str, env: dict | None = None, expect: int = 0, timeout: int = 300) -> str:
        cmd = [sys.executable, *args]
        p = subprocess.run(cmd, cwd=self.work, env={**self.env, **(env or {})}, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out = p.stdout + p.stderr
        if p.returncode != expect:
            tail = "\n".join(out.strip().splitlines()[-25:])
            raise StageFailed(f"`python {' '.join(args)}` exited {p.returncode} (expected {expect}):\n{tail}")
        return out

    def stage(self, name: str, fn) -> bool:
        start = time.monotonic()
        try:
            note = fn() or ""
            ok, err = True, ""
        except StageFailed as e:
            ok, err, note = False, str(e), ""
        except Exception:
            ok, err, note = False, traceback.format_exc(limit=6), ""
        secs = round(time.monotonic() - start, 1)
        self.results.append({"stage": name, "ok": ok, "seconds": secs, "note": note, "error": err})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({secs}s) {note}", flush=True)
        if err:
            print("  " + err.replace("\n", "\n  "), flush=True)
        return ok

    # -- stages -----------------------------------------------------------------------------------
    def setup(self) -> str:
        if self.work.exists():
            for child in self.work.iterdir():  # keep the dir itself: it may be a mounted volume
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        self.work.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO / "configs", self.work / "configs")
        (self.work / "evals").mkdir()
        shutil.copy(REPO / "evals" / "adversarial_test.jsonl", self.work / "evals")
        src = self.data_root / "corpus"
        check(src.is_dir() and any(src.glob("*.jsonl")), f"no corpus at {src} (mount the main checkout's corpus/)")
        shutil.copytree(src, self.work / "corpus")
        pdfs = list((self.data_root / "sources" / "pdf").glob("*.pdf"))
        idx = sorted(p.name for p in (self.data_root / "index").glob("*") if p.is_dir())
        arms = (self.work / "configs" / "arms.yaml")
        arms.write_text(arms.read_text(encoding="utf-8").replace("n_rows: 3000", f"n_rows: {N_ARM_ROWS}"),
                        encoding="utf-8")
        check(f"n_rows: {N_ARM_ROWS}" in arms.read_text(encoding="utf-8"), "could not shrink n_rows in arms.yaml")
        return f"{len(list(src.glob('*.jsonl')))} corpus docs, {len(pdfs)} source pdfs, mounted indexes {idx}"

    def split(self) -> str:
        self.run("-m", "pipeline.split", "corpus", "data/splits.json")
        self.splits = json.loads((self.work / "data/splits.json").read_text(encoding="utf-8"))
        check(set(self.splits) == {"train", "dev", "test"}, f"split keys {set(self.splits)}")
        docs = [d for s in self.splits.values() for d in s]
        check(len(docs) == len(set(docs)), "a document is in two splits")
        corpus = {p.stem for p in (self.work / "corpus").glob("*.jsonl")}
        check(set(docs) == corpus, f"split docs != corpus docs: {sorted(set(docs) ^ corpus)}")
        check(all(self.splits[s] for s in self.splits), "an empty split")
        self.run("-m", "pipeline.split", "corpus", "data/splits2.json")
        check((self.work / "data/splits2.json").read_bytes() == (self.work / "data/splits.json").read_bytes(),
              "split is not deterministic")
        main = self.data_root / "data" / "splits.json"
        same = "n/a" if not main.exists() else json.loads(main.read_text(encoding="utf-8")) == self.splits
        return f"{ {k: len(v) for k, v in self.splits.items()} }, matches main data/splits.json: {same}"

    def build_index(self) -> str:
        self.run("-m", "retrieval.build_index", "corpus", "index", "--version", INDEX_VERSION)
        idx = self.work / "index" / INDEX_VERSION
        n_corpus = sum(1 for f in (self.work / "corpus").glob("*.jsonl") for _ in f.open(encoding="utf-8"))
        passages = jsonl(idx / "passages.jsonl")
        meta = json.loads((idx / "meta.json").read_text(encoding="utf-8"))
        check(len(passages) == n_corpus == meta["passages"], f"index {len(passages)} vs corpus {n_corpus}")
        check(meta["index_version"] == INDEX_VERSION, f"meta index_version {meta['index_version']}")
        check(len({p["pid"] for p in passages}) == len(passages), "duplicate pids in index")
        out = self.run("-m", "retrieval.build_index", "corpus", "index", "--version", INDEX_VERSION)
        check("up to date" in out, "rebuild with unchanged inputs was not skipped")
        return f"{len(passages)} passages, idempotent rebuild"

    def generate(self) -> str:
        from tests.e2e.fake_generate import SMALL_TARGETS

        self.run("-m", "tests.e2e.fake_generate", "--index", f"index/{INDEX_VERSION}")
        doc_split = {d: s for s, ds in self.splits.items() for d in ds}
        pid_doc = {p["pid"]: p["doc_id"] for p in jsonl(self.work / "index" / INDEX_VERSION / "passages.jsonl")}
        counts = {}
        for split, target in SMALL_TARGETS.items():
            rows = jsonl(self.work / "data/raw" / f"{split}.jsonl")
            counts[split] = len(rows)
            check(len(rows) == target, f"{split}: {len(rows)} raw rows, expected {target}")
            for r in rows:
                TrainingRow.model_validate({k: v for k, v in r.items() if k != "job_key"})
                check(r["split"] == split and doc_split[r["provenance"]["source_doc"]] == split,
                      f"{r['row_id']} source doc not in {split}")
                for p in r["passages"]:
                    check(doc_split[pid_doc[p["pid"]]] == split,
                          f"{r['row_id']} retrieved passage {p['pid']} from another split (leak)")
        out = self.run("-m", "tests.e2e.fake_generate", "--index", f"index/{INDEX_VERSION}")
        check("0 jobs to run" in out, "rerun was not resumable (jobs re-ran)")
        kinds = Counter(r["kind"] for r in jsonl(self.work / "data/raw/train.jsonl"))
        return f"raw {counts}, train kinds {dict(kinds)}, resume ok"

    def curate(self) -> str:
        self.run("-m", "pipeline.curate", "--raw", "data/raw", "--corpus", "corpus", "--out", "data/curated",
                 "--no-llm-gates")
        out = {}
        for split in ("train", "dev", "test"):
            raw = jsonl(self.work / "data/raw" / f"{split}.jsonl")
            cur = jsonl(self.work / "data/curated" / f"{split}.jsonl")
            out[split] = f"{len(raw)}->{len(cur)}"
            check(cur, f"{split}: 0 curated rows")
            # Fake-teacher rows are built to pass every gate; a big drop means a gate regressed.
            check(len(cur) >= 0.7 * len(raw), f"{split}: only {len(cur)}/{len(raw)} rows survived curation")
            raw_ids = {r["row_id"] for r in raw}
            for r in cur:
                check(r["row_id"] in raw_ids, f"curated row {r['row_id']} not in raw")
                TrainingRow.model_validate({k: v for k, v in r.items() if k != "job_key"})
                check(r["provenance"].get("gates") is not None, f"{r['row_id']} has no gate record")
        log = jsonl(self.work / "data/curated/gate_log.jsonl")
        fails = Counter((g["gate"], g["reason"][:40]) for g in log if g["decision"] == "fail")
        check(log, "empty gate log")
        return f"{out}; gate fails {dict(fails.most_common(5))}"

    def build_arms(self) -> str:
        self.run("-m", "pipeline.build_arms", "arms")
        a1 = jsonl(self.work / "data/arms/A1/rows.jsonl")
        a2 = jsonl(self.work / "data/arms/A2/rows.jsonl")
        check(len(a1) == len(a2) == N_ARM_ROWS, f"A1={len(a1)} A2={len(a2)}, expected {N_ARM_ROWS}")
        strata = lambda rows: Counter((r["lang"], r["answerable"]) for r in rows)  # noqa: E731
        check(strata(a1) == strata(a2), f"A1/A2 strata differ: {strata(a1)} vs {strata(a2)}")
        curated_ids = {r["row_id"] for r in jsonl(self.work / "data/curated/train.jsonl")}
        check(all(r["row_id"] in curated_ids for r in a2), "A2 contains a non-curated row")
        for arm in ("A1", "A2"):
            sft = jsonl(self.work / "data/arms" / arm / "train.jsonl")
            check(len(sft) == N_ARM_ROWS, f"{arm} train.jsonl has {len(sft)} rows")
            for r in sft:
                roles = [m["role"] for m in r["messages"]]
                check(roles == ["user", "assistant"], f"{arm} {r['row_id']} roles {roles}")
                ModelOutput.model_validate_json(r["messages"][1]["content"])
                check("<question>" in r["messages"][0]["content"], f"{arm} prompt is not the shared template")
        dev = jsonl(self.work / "data/arms/dev.jsonl")
        check(dev, "empty dev set")
        check_no_overlap(a2 + dev + jsonl(self.work / "data/curated/test.jsonl"))
        doc_split = {d: s for s, ds in self.splits.items() for d in ds}
        for rows, split in ((a1 + a2, "train"), (dev, "dev")):
            bad = [r["row_id"] for r in rows if doc_split[r["provenance"]["source_doc"]] != split]
            check(not bad, f"{split} rows from another split's docs (leak): {bad[:3]}")
        return f"A1=A2={N_ARM_ROWS}, strata {dict(strata(a2))}, dev={len(dev)}, leakage check ok"

    def review_freeze(self) -> str:
        self.run("-m", "pipeline.build_arms", "review")
        review = list(csv.DictReader((self.work / "data/test/review.csv").open(encoding="utf-8-sig")))
        cur_test = jsonl(self.work / "data/curated/test.jsonl")
        check(len(review) == len(cur_test), f"review.csv {len(review)} rows vs curated test {len(cur_test)}")
        self.run("-m", "pipeline.build_arms", "freeze", "--index", f"index/{INDEX_VERSION}", expect=1)
        self.run("-m", "pipeline.build_arms", "freeze", "--index", f"index/{INDEX_VERSION}", "--allow-unreviewed")
        test = self.work / "data/test/test.jsonl"
        digest = (self.work / "data/test/test.sha256").read_text().strip()
        check(hashlib.sha256(test.read_bytes()).hexdigest() == digest, "test.sha256 does not match test.jsonl")
        rows = jsonl(test)
        n_adv = len(jsonl(self.work / "evals/adversarial_test.jsonl"))
        kinds = Counter(r["kind"] for r in rows)
        check(kinds["adversarial"] == n_adv, f"{kinds['adversarial']} adversarial rows, expected {n_adv}")
        check(kinds["answerable"] > 0 and kinds["unanswerable"] > 0, f"test kinds {dict(kinds)}")
        check(all(1 <= len(r["passages"]) <= 4 for r in rows), "a frozen test row has 0 or >4 passages")
        return f"froze {len(rows)} rows {dict(kinds)}, sha {digest[:12]}, unreviewed freeze refused"

    # -- API --------------------------------------------------------------------------------------
    def api(self) -> str:
        import httpx

        port = free_port()
        env = {**self.env, "DHANGROUND_GENERATOR": FAKE_GEN, "INDEX_DIR": f"index/{INDEX_VERSION}",
               "DEMO_MODE": "gpu"}
        log = (self.work / "api.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1",
                                 "--port", str(port)], cwd=self.work, env=env, stdout=log, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        try:
            health = wait_health(base, proc)
            check(health["mode"] == "gpu", f"health mode {health['mode']} (fake generator not loaded?)")
            check(health["index_version"] == INDEX_VERSION, f"health index_version {health['index_version']}")
            with httpx.Client(base_url=base, timeout=30) as c:
                notes = self.api_contract(c)
            notes += "; " + self.warmup(base)
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        return notes

    def api_contract(self, c) -> str:
        h = {"x-session-id": "e2e-1"}
        r = c.post("/v1/ask", json={"request_id": "e2e-ask-1", "arm": "A2",
                                    "question": "UPI payment fail hua, account se paisa kat gaya, kab wapas aayega?"},
                   headers=h)
        check(r.status_code == 200, f"/v1/ask {r.status_code}: {r.text[:300]}")
        body = r.json()
        check(API_FIELDS <= set(body), f"/v1/ask missing fields {API_FIELDS - set(body)}")
        check(body["status"] == "ok" and body["request_id"] == "e2e-ask-1" and body["arm"] == "A2",
              f"/v1/ask body {json.dumps(body, ensure_ascii=False)[:400]}")
        check(body["citations"] and {"pid", "title", "url", "snippet"} <= set(body["citations"][0]),
              f"citations {body['citations']}")
        check(set(body["checks"]) == {"schema", "citations", "claim_check"}, f"checks {body['checks']}")
        check(body["checks"]["schema"] == "pass", f"checks {body['checks']}")
        check(body["index_version"] == INDEX_VERSION and body["source"] == "live", "index_version/source")

        r2 = c.post("/v1/ask", json={"request_id": "e2e-ask-1", "question": "something else", "arm": "A2"},
                    headers=h)
        check(r2.json() == body, "request_id dedupe returned a different body")

        r = c.post("/v1/ask", json={"request_id": "e2e-abs", "arm": "A3",
                                    "question": "Which bank gives the best rate today for a fixed deposit?"},
                   headers=h)
        check(r.status_code == 200 and r.json()["status"] == "abstained", f"abstain path: {r.text[:300]}")

        r = c.post("/v1/ask", json={"request_id": "e2e-bad", "question": "", "arm": "A2"}, headers=h)
        err = r.json()
        check(r.status_code == 400 and err["status"] == "error" and err["error"]["code"] == "BAD_INPUT"
              and err["request_id"] == "e2e-bad", f"BAD_INPUT shape: {r.status_code} {r.text[:200]}")

        r = c.post("/v1/compare", json={"request_id": "e2e-cmp", "question": "ATM se paisa nahi nikla, kya karun?"},
                   headers={"x-session-id": "e2e-2"})
        check(r.status_code == 200, f"/v1/compare {r.status_code}: {r.text[:300]}")
        arms = [x["arm"] for x in r.json()["results"]]
        check(arms == ["A0", "A1", "A2", "A3"], f"/v1/compare arms {arms}")

        ex = c.get("/v1/examples").json()
        check(isinstance(ex.get("examples"), list) and ex["examples"], "/v1/examples empty")

        for i in range(12):
            r = c.post("/v1/ask", json={"request_id": f"e2e-rl-{i}", "question": "UPI refund kab aayega?"},
                       headers={"x-session-id": "e2e-rl"})
        err = r.json()
        check(r.status_code == 429 and err["error"]["code"] == "RATE_LIMITED" and err["error"]["retry_after_s"] > 0,
              f"rate limit: {r.status_code} {r.text[:200]}")
        return f"ask ok (lang {body['lang']}, claim_check {body['checks']['claim_check']}), dedupe, abstain, " \
               f"BAD_INPUT, compare x4, examples {len(ex['examples'])}, RATE_LIMITED"

    def warmup(self, base: str) -> str:
        script = REPO / "scripts" / "warmup.py"
        if not script.exists():
            return "warmup: SKIP (scripts/warmup.py not on this branch)"
        out = self.run(str(script), "--url", base, timeout=180)
        return f"warmup ok: {out.strip().splitlines()[-1][:120] if out.strip() else ''}"

    # -- eval -------------------------------------------------------------------------------------
    def evals(self) -> str:
        env = {"DHANGROUND_GENERATOR": FAKE_GEN, "INDEX_DIR": f"index/{INDEX_VERSION}"}
        self.run("-m", "evals.run_eval", "--arm", "A2", "--seed", "0", "--split", "dev", "--no-judge", env=env)
        res = json.loads((self.work / "evals/results/A2_s0_dev.json").read_text(encoding="utf-8"))
        dev = jsonl(self.work / "data/arms/dev.jsonl")
        verdicts = jsonl(self.work / "evals/results/A2_s0_dev_verdicts.jsonl")
        check(res["n"] == len(dev) == len(verdicts), f"eval n={res['n']}, dev={len(dev)}, verdicts={len(verdicts)}")
        check(res["metrics"] and all("value" in m for m in res["metrics"].values()), "metrics missing values")
        check(res["slices"], "no slices")
        ok = sum(v["status"] == "ok" for v in verdicts)
        check(ok > 0, "fake generator produced no ok answers on dev")

        self.run("-m", "evals.run_eval", "--arm", "A0", "--seed", "0", "--split", "test", "--no-judge", env=env)
        out = self.run("-m", "evals.run_eval", "--arm", "A0", "--seed", "0", "--split", "test", "--no-judge",
                       env=env, expect=1)
        check("used once" in out, "second test scoring was not refused")
        test = self.work / "data/test/test.jsonl"
        good = test.read_bytes()
        test.write_bytes(good + b"\n")
        try:
            out = self.run("-m", "evals.run_eval", "--arm", "A1", "--seed", "0", "--split", "test", "--no-judge",
                           env=env, expect=1)
            check("sha256" in out, "tampered test.jsonl was not refused")
        finally:
            test.write_bytes(good)
        m = {k: v["value"] for k, v in res["metrics"].items()}
        return f"dev n={res['n']} ok={ok} metrics {json.dumps(m)[:200]}; test once-only + sha guard ok"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_health(base: str, proc, timeout: float = 60) -> dict:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise StageFailed(f"API exited with {proc.returncode}; see api.log")
        try:
            r = httpx.get(base + "/v1/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise StageFailed("API did not become healthy in 60 s")


def main() -> None:
    default_root = Path(os.environ.get("DHANGROUND_DATA", str(REPO)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=default_root)
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("E2E_WORK", REPO / ".e2e-work")))
    a = ap.parse_args()
    h = Harness(a.data_root.resolve(), a.work.resolve())
    t0 = time.monotonic()
    print(f"e2e: repo {REPO}, data {h.data_root}, work {h.work}, python {sys.version.split()[0]}", flush=True)
    stages = [("setup", h.setup), ("pipeline.split", h.split), ("retrieval.build_index", h.build_index),
              ("intelligence.generate (fake teacher)", h.generate), ("pipeline.curate --no-llm-gates", h.curate),
              ("pipeline.build_arms arms", h.build_arms), ("build_arms review/freeze", h.review_freeze),
              ("apps.api (fake generator) + warmup", h.api), ("evals.run_eval", h.evals)]
    for name, fn in stages:
        if not h.stage(name, fn):
            break  # later stages need this one's outputs
    total = round(time.monotonic() - t0, 1)
    ok = len(h.results) == len(stages) and all(r["ok"] for r in h.results)
    (h.work / "e2e_report.json").write_text(json.dumps({"ok": ok, "seconds": total, "stages": h.results},
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"E2E {'PASS' if ok else 'FAIL'} in {total}s", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
