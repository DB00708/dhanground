"""Publish DhanGround's datasets, results and adapter collection to the Hugging Face Hub.

Usage (from the repo root):
  python scripts/publish_hf.py --org <org>                     # dry run: repo ids, files, rows, sizes
  python scripts/publish_hf.py --org <org> --push              # create/update repos (private)
  python scripts/publish_hf.py --org <org> --push --public     # same, and make every repo public

Three dataset repos, each with a generated dataset card (README.md):
  <org>/dhanground-rbi-corpus   corpus/*.jsonl
  <org>/dhanground-hinglish-qa  data/curated/{train,dev,test}.jsonl, gate_log.jsonl, the frozen test set and its
                                sha256, evals/test_set_card.md, evals/results/* under results/
  <org>/dhanground-dpo-pairs    data/arms/A3_pairs.jsonl
and two collections under <org>: "DhanGround datasets" (the three datasets; the gate log, frozen test set and
eval results live in dhanground-hinglish-qa) and "DhanGround models" (the base model from configs/arms.yaml plus
every model repo <org>/dhanground-*, i.e. the adapters that adapters.aligntune_sft / aligntune_dpo pushed).

The script refuses (exit 2, nothing uploaded) if any input is missing, if data/test/test.jsonl does not match
data/test/test.sha256, or if evals/test_set_card.md is still the placeholder.
huggingface_hub calls used (checked against huggingface_hub 2.4.0 source): HfApi.create_repo, HfApi.upload_folder,
HfApi.update_repo_settings, HfApi.list_models, HfApi.create_collection, HfApi.update_collection_metadata,
HfApi.add_collection_item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The submission needs two public collections: models (base + adapters) and datasets (data, logs, results).
MODEL_COLLECTION = ("DhanGround models",
                    "Tiny Aya Fire base model and DhanGround LoRA adapters (arms A1, A2, A3) for RBI-grounded banking QA.")
DATASET_COLLECTION = ("DhanGround datasets",
                      "RBI corpus, Hinglish banking QA with gate log, frozen test set, eval results, DPO pairs.")
PLACEHOLDER_MARK = "<!-- placeholder: run python -m evals.make_test_card -->"  # same as evals/make_test_card.py

LICENCE = """\
- Corpus text: public documents from the Reserve Bank of India (rbi.org.in), copied as published, one row per
  paragraph, with the source URL and date on every row. RBI text is public; check rbi.org.in for its terms
  before any reuse beyond research.
- Questions, answers and pairs: written by a teacher LLM from that text, then curated and partly hand-checked.
  Use them for research only. They are not financial or legal advice.
- Model: the adapters in the DhanGround collection are LoRA adapters for Cohere Tiny Aya. Tiny Aya weights are
  released for non-commercial use only, so the adapters and any outputs from them are non-commercial too."""

FIELD_DOCS = {
    # corpus rows
    "pid": "passage id, `{doc_id}-{para_id}`; answers cite these",
    "doc_id": "RBI document id",
    "para_id": "paragraph number within the document (`p1`, `p2`, ...)",
    "text": "paragraph text as published",
    "sha256": "sha256 of `text`",
    "title": "document title",
    "url": "source URL on rbi.org.in",
    "date": "document date",
    "topics": "topic tags from configs/rbi_sources.yaml",
    # QA rows
    "row_id": "stable row id",
    "lang": "language of question and answer: `hi-Deva`, `hi-Latn` (Hinglish) or `en`",
    "question": "user question",
    "passages": "retrieved passages shown to the model (`pid`, `text`, `score`)",
    "answerable": "true if the passages answer the question; false means the model must abstain",
    "answer": "gold answer with citations, or the abstain message",
    "cited_pids": "passage ids the gold answer cites",
    "numbers": "numeric claims in the answer (days, amounts) for the claim check",
    "provenance": "`source_doc`, `span_sha256` of the source paragraph, `generator` model",
    "split": "train, dev or test (split by document, stratified by topic)",
    "kind": "`answerable`, `unanswerable` or `adversarial`",
    "source_pid": "paragraph the question was generated from",
    "job_key": "generation job key (resume key)",
    "attack_type": "adversarial rows only: the attack (e.g. `injection`, `out_of_corpus`)",
    # gate log
    "gate": "curation gate name",
    "decision": "`pass` or `fail`",
    "reason": "why the gate decided so",
    "score": "gate score, if the gate has one",
    # DPO pairs
    "prompt": "chat messages: the rendered user turn (question + passages)",
    "chosen": "assistant message: the curated gold answer",
    "rejected": "assistant message: the A2 adapter's failed answer",
    "reject_reason": "why the A2 answer failed (schema, claim check, abstain error, judge)",
}


@dataclass
class Item:
    local: Path
    path_in_repo: str

    @property
    def size(self) -> int:
        return self.local.stat().st_size

    @property
    def rows(self) -> int | None:
        if self.local.suffix != ".jsonl":
            return None
        with self.local.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


@dataclass
class DatasetRepo:
    name: str
    title: str
    summary: str
    provenance: list[str]
    items: list[Item] = field(default_factory=list)
    card: str = ""

    def repo_id(self, org: str) -> str:
        return f"{org}/{self.name}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def first_rows_keys(paths: list[Path]) -> list[str]:
    keys: dict[str, None] = {}
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                if line.strip():
                    keys.update(dict.fromkeys(json.loads(line)))
    return list(keys)


def check_inputs(root: Path) -> list[str]:
    """Every problem that blocks publishing; empty means ready."""
    problems = []
    required = [
        "corpus/state.json", "data/curated/train.jsonl", "data/curated/dev.jsonl", "data/curated/test.jsonl",
        "data/curated/gate_log.jsonl", "data/test/test.jsonl", "data/test/test.sha256",
        "data/arms/A3_pairs.jsonl", "evals/test_set_card.md", "configs/arms.yaml",
    ]
    problems += [f"missing {p}" for p in required if not (root / p).is_file()]
    if not list((root / "corpus").glob("*.jsonl")):
        problems.append("missing corpus/*.jsonl (run python -m pipeline.ingest)")
    results = root / "evals" / "results"
    if not results.is_dir() or not any(p.is_file() for p in results.iterdir()):
        problems.append("missing evals/results/* (run the evaluation stage first)")
    test, sha = root / "data/test/test.jsonl", root / "data/test/test.sha256"
    if test.is_file() and sha.is_file():
        want = sha.read_text(encoding="utf-8").strip()
        got = sha256_file(test)
        if got != want:
            problems.append(f"data/test/test.jsonl sha256 {got} does not match test.sha256 {want}")
    card = root / "evals/test_set_card.md"
    if card.is_file():
        text = card.read_text(encoding="utf-8")
        if PLACEHOLDER_MARK in text:
            problems.append("evals/test_set_card.md is the placeholder (run python -m evals.make_test_card)")
        elif test.is_file() and sha256_file(test) not in text:
            problems.append("evals/test_set_card.md is stale: it does not name the frozen test sha256 "
                            "(run python -m evals.make_test_card)")
    for p in [*sorted((root / "corpus").glob("*.jsonl")), *(root / x for x in required if x.endswith(".jsonl"))]:
        if p.is_file() and p.stat().st_size == 0:
            problems.append(f"{p.relative_to(root).as_posix()} is empty")
    return problems


def snapshot_date(root: Path) -> str:
    manifest = json.loads((root / "corpus/state.json").read_text(encoding="utf-8"))
    return manifest["written_at"][:10]


def training_logs(root: Path) -> list[Item]:
    """Per local training run: its dhanground.json and the newest checkpoint's trainer_state.json (optional)."""
    items = []
    for rec in sorted([*(root / "checkpoints").glob("*/dhanground.json"),
                       *(root / "checkpoints").glob("*/final/dhanground.json")]):
        run = rec.parent if rec.parent.name != "final" else rec.parent.parent
        items.append(Item(rec, f"training_logs/{run.name}/dhanground.json"))
        states = sorted(run.glob("checkpoint-*/trainer_state.json"),
                        key=lambda q: int(q.parent.name.split("-")[-1]) if q.parent.name.split("-")[-1].isdigit() else -1)
        if states:
            items.append(Item(states[-1], f"training_logs/{run.name}/trainer_state.json"))
    return items


def build_repos(root: Path) -> list[DatasetRepo]:
    corpus = DatasetRepo(
        "dhanground-rbi-corpus", "DhanGround RBI corpus",
        "Public Reserve Bank of India circulars, master directions and FAQs on consumer banking, one row per "
        "paragraph. This is the only knowledge source the DhanGround model may answer from.",
        ["`python -m pipeline.ingest configs/rbi_sources.yaml corpus/` fetched each allowlisted rbi.org.in page "
         "(PDFs saved by hand: the PDF host shows a CAPTCHA to scripts) and split it into paragraphs.",
         "One file per document: `corpus/{doc_id}.jsonl`."],
        [Item(p, f"corpus/{p.name}") for p in sorted((root / "corpus").glob("*.jsonl"))],
    )
    qa = DatasetRepo(
        "dhanground-hinglish-qa", "DhanGround Hinglish banking QA",
        "Questions about Indian banking rules in Hindi (Devanagari), Hinglish (romanized Hindi) and English, each "
        "with retrieved RBI passages and a cited answer, or an abstain label when the passages do not answer it. "
        "Includes the curation log, the frozen test set and the evaluation results.",
        ["Generate: a teacher LLM wrote questions from corpus paragraphs (`intelligence/generate.py`).",
         "Curate: deterministic gates and CuratorKIT dedup on every row; CuratorKIT grounding and reward gates on "
         "answerable rows only when a judge model was set (`pipeline/curate.py`; `provenance.gates.halluc` is "
         "`pass` or `n/a` per row). `gate_log.jsonl` has one decision per row and gate.",
         "Test: hand-checked, sampled and frozen with hand-written adversarial rows "
         "(`pipeline/build_arms.py freeze`); see `test_set_card.md`. `frozen_test/test.sha256` pins the file; "
         "the evaluator refuses a mismatch and scores each (arm, seed) on test once.",
         "Results: `results/` holds per-(arm, seed) metrics with bootstrap CIs, per-row verdicts (eval traces), "
         "the summary, general capability and ROI.",
         "Training logs: `training_logs/<run>/` holds each local run's `dhanground.json` (arm, seed, base, backend) "
         "and the `trainer_state.json` (loss log) of its newest checkpoint, when present."],
        [Item(root / f"data/curated/{s}.jsonl", f"data/{s}.jsonl") for s in ("train", "dev", "test")]
        + [Item(root / "data/curated/gate_log.jsonl", "data/gate_log.jsonl"),
           Item(root / "data/test/test.jsonl", "frozen_test/test.jsonl"),
           Item(root / "data/test/test.sha256", "frozen_test/test.sha256"),
           Item(root / "evals/test_set_card.md", "test_set_card.md")]
        + [Item(p, f"results/{p.relative_to(root / 'evals/results').as_posix()}")
           for p in sorted((root / "evals/results").rglob("*")) if p.is_file()]
        + training_logs(root),
    )
    pairs = DatasetRepo(
        "dhanground-dpo-pairs", "DhanGround DPO pairs",
        "Preference pairs mined from the A2 (curated SFT) adapter's own failures on dev and unused curated train "
        "rows: chosen = curated gold answer, rejected = the adapter's failed answer. Used to train arm A3 with DPO.",
        ["`python -m intelligence.mine_pairs --a2 <A2 adapter> --extra-curated-train 1500` ran the A2 adapter and "
         "kept answers that failed the schema check, the claim check, the abstain decision or the judge."],
        [Item(root / "data/arms/A3_pairs.jsonl", "data/A3_pairs.jsonl")],
    )
    return [corpus, qa, pairs]


def render_card(repo: DatasetRepo, org: str, snapshot: str) -> str:
    jsonl = [i for i in repo.items if i.local.suffix == ".jsonl"]
    groups: dict[str, list[Path]] = {}
    for i in jsonl:
        key = "gate log" if i.local.name == "gate_log.jsonl" else "rows"
        groups.setdefault(key, []).append(i.local)
    lines = [
        "---",
        f"pretty_name: {repo.title}",
        "language: [hi, en]",
        "license: other",
        "task_categories: [question-answering]",
        "tags: [rbi, banking, hinglish, rag, grounded-qa, dhanground]",
        "---",
        "",
        f"# {repo.title}",
        "",
        repo.summary,
        "",
        f"Part of the DhanGround collection by [{org}](https://huggingface.co/{org}). Corpus snapshot date: "
        f"{snapshot} (when the RBI pages were fetched).",
        "",
        "## Files",
        "",
        "| File | Rows | Size |",
        "|---|---:|---:|",
        *[f"| `{i.path_in_repo}` | {i.rows if i.rows is not None else '-'} | {human(i.size)} |" for i in repo.items],
        "",
        "## Fields",
        "",
    ]
    for key, paths in groups.items():
        if len(groups) > 1:
            lines += [f"**{key}**", ""]
        lines += [f"- `{k}`: {FIELD_DOCS.get(k, 'see the DhanGround source code')}" for k in first_rows_keys(paths)]
        lines.append("")
    lines += ["## Provenance", "", *[f"{n}. {p}" for n, p in enumerate(repo.provenance, 1)], "",
              "## Licence", "", LICENCE, ""]
    return "\n".join(lines)


def local_adapter_repos(root: Path, org: str) -> list[str]:
    """Adapter repos the training scripts would have pushed, from local checkpoints/ records."""
    names = set()
    for rec in (root / "checkpoints").glob("*/dhanground.json"):
        names.add(rec.parent.name)
    for rec in (root / "checkpoints").glob("*/final/dhanground.json"):
        names.add(rec.parent.parent.name)
    return [f"{org}/dhanground-{n}" for n in sorted(names)]


def print_plan(repos: list[DatasetRepo], org: str, public: bool, root: Path) -> None:
    vis = "public" if public else "private"
    for r in repos:
        total = sum(i.size for i in r.items)
        print(f"dataset {r.repo_id(org)} ({vis}): {len(r.items)} files + README.md, {human(total)}")
        for i in r.items:
            rows = f"{i.rows} rows" if i.rows is not None else "-"
            print(f"  {i.path_in_repo:<40} {rows:>12} {human(i.size):>10}")
    adapters = local_adapter_repos(root, org)
    print(f'collection "{DATASET_COLLECTION[0]}" ({vis}) under {org}: the {len(repos)} datasets')
    print(f'collection "{MODEL_COLLECTION[0]}" ({vis}) under {org}: base model {base_model(root)} + model repos '
          f"{org}/dhanground-* found on the Hub at push time")
    print("  adapters expected from local checkpoints/: " + (", ".join(adapters) if adapters else "(none local)"))


def base_model(root: Path) -> str:
    import yaml

    return yaml.safe_load((root / "configs/arms.yaml").read_text(encoding="utf-8"))["base_model"]


def collection(api, title_desc: tuple[str, str], org: str, public: bool):
    title, description = title_desc
    coll = api.create_collection(title, namespace=org, description=description, private=not public, exists_ok=True)
    if public:  # an existing collection keeps its old visibility
        api.update_collection_metadata(coll.slug, private=False)
    return coll


def push(repos: list[DatasetRepo], org: str, public: bool, base: str) -> tuple[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    for r in repos:
        repo_id = r.repo_id(org)
        api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
        if public:  # create_repo ignores `private` for an existing repo
            api.update_repo_settings(repo_id, repo_type="dataset", private=False)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            for i in r.items:
                dest = stage / i.path_in_repo
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(i.local, dest)
            (stage / "README.md").write_text(r.card, encoding="utf-8")
            # delete_patterns="*" drops files from an earlier publish that this one no longer has, so the repo
            # matches the card's Files table (upload_folder keeps .gitattributes and the files being uploaded)
            api.upload_folder(repo_id=repo_id, folder_path=str(stage), repo_type="dataset",
                              commit_message="Publish DhanGround data", delete_patterns="*")
        print(f"pushed {repo_id}")

    datasets = collection(api, DATASET_COLLECTION, org, public)
    for r in repos:
        api.add_collection_item(datasets.slug, r.repo_id(org), "dataset", exists_ok=True)
    print(f"collection {datasets.slug}: {len(repos)} datasets")

    models = collection(api, MODEL_COLLECTION, org, public)
    api.add_collection_item(models.slug, base, "model", note="Base model (gated; non-commercial licence)",
                            exists_ok=True)
    prefix = f"{org}/dhanground-"
    adapters = sorted(m.id for m in api.list_models(author=org, search="dhanground-") if m.id.startswith(prefix))
    for repo_id in adapters:
        if public:
            api.update_repo_settings(repo_id, repo_type="model", private=False)
        api.add_collection_item(models.slug, repo_id, "model", exists_ok=True)
    print(f"collection {models.slug}: base model + {len(adapters)} adapters")
    if not adapters:
        print("WARNING: no <org>/dhanground-* adapter repos on the Hub; train with --hub-org to push them")
    return datasets.slug, models.slug


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--org", required=True, help="HF user or organization that owns the repos")
    ap.add_argument("--push", action="store_true", help="upload; without it this is a dry run")
    ap.add_argument("--public", action="store_true", help="make the repos and collection public")
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root holding corpus/, data/, evals/")
    a = ap.parse_args(argv)

    problems = check_inputs(a.root)
    if problems:
        print("refusing to publish:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    repos = build_repos(a.root)
    snapshot = snapshot_date(a.root)
    for r in repos:
        r.card = render_card(r, a.org, snapshot)
    print_plan(repos, a.org, a.public, a.root)
    if not a.push:
        print("dry run: nothing uploaded (add --push)")
        return 0
    push(repos, a.org, a.public, base_model(a.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
