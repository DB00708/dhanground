"""scripts/publish_hf.py and evals/make_test_card.py on a fake repo tree; huggingface_hub is a fake module.

The fake HfApi methods copy the parameter lists of huggingface_hub 2.4.0 (hf_api.py), so a wrong argument
name or a positional where the real API wants a keyword fails here. Nothing calls Hugging Face.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from evals import make_test_card
from scripts import publish_hf


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def qa_row(i, kind="answerable", lang="hi-Latn", doc="rbi-1", **extra):
    return {"row_id": f"dg-{i}", "lang": lang, "question": f"q{i}", "passages": [], "answerable": kind == "answerable",
            "answer": "a", "cited_pids": [], "provenance": {"source_doc": doc, "span_sha256": "x", "generator": "t"},
            "split": "test", "kind": kind, **extra}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path
    write_jsonl(root / "corpus/rbi-1.jsonl", [{"pid": "rbi-1-p1", "doc_id": "rbi-1", "text": "t", "topics": ["upi"]}])
    write_jsonl(root / "corpus/faq-2.jsonl", [{"pid": "faq-2-p1", "doc_id": "faq-2", "text": "t", "topics": ["atm"]}] * 2)
    (root / "corpus/state.json").write_text(json.dumps({"state": "INGESTED", "written_at": "2026-11-10T06:25:13+00:00"}))
    for s in ("train", "dev", "test"):
        write_jsonl(root / f"data/curated/{s}.jsonl", [qa_row(1), qa_row(2)])
    write_jsonl(root / "data/curated/gate_log.jsonl", [{"row_id": "dg-1", "split": "train", "gate": "dedup",
                                                        "decision": "pass", "reason": "", "score": None}])
    test = [qa_row(1), qa_row(2, kind="unanswerable", lang="en", doc="faq-2"),
            qa_row(3, kind="adversarial", lang="hi-Deva", doc="", attack_type="injection")]
    write_jsonl(root / "data/test/test.jsonl", test)
    digest = hashlib.sha256((root / "data/test/test.jsonl").read_bytes()).hexdigest()
    (root / "data/test/test.sha256").write_text(digest + "\n")
    (root / "data/test/review.csv").write_text("row_id,keep\ndg-1,y\ndg-2,y\ndg-9,n\n", encoding="utf-8")
    write_jsonl(root / "data/arms/A3_pairs.jsonl", [{"row_id": "dg-1", "prompt": [], "chosen": [], "rejected": [],
                                                     "reject_reason": "schema"}])
    (root / "evals/results").mkdir(parents=True)
    (root / "evals/results/A2_s0_test.json").write_text("{}")
    (root / "evals/results/summary_test.md").write_text("# s")
    (root / "evals/test_set_card.md").write_text(
        make_test_card.build_card(root / "data/test/test.jsonl", root / "data/test/review.csv", root / "corpus"),
        encoding="utf-8")
    for name in ("A2-s0-abc1234", "A1-s1-abc1234"):
        (root / "checkpoints" / name).mkdir(parents=True)
        (root / "checkpoints" / name / "dhanground.json").write_text("{}")
    (root / "checkpoints/A3-s0-abc1234/final").mkdir(parents=True)
    (root / "checkpoints/A3-s0-abc1234/final/dhanground.json").write_text("{}")
    return root


class FakeHub:
    def __init__(self, models=()):
        self.calls, self.uploads, self.models = [], {}, list(models)
        hub = self

        class Collection:
            slug = "org/dhanground-64f9a55bb3115b4f513ec026"

        class HfApi:
            def __init__(self, endpoint=None, token=None, library_name=None, library_version=None,
                         user_agent=None, headers=None):
                pass

            def create_repo(self, repo_id, *, token=None, private=None, visibility=None, repo_type=None,
                            exist_ok=False, resource_group_id=None, region=None, space_sdk=None,
                            space_hardware=None, space_sleep_time=None, space_secrets=None, space_variables=None,
                            space_volumes=None, space_template=None):
                hub.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

            def update_repo_settings(self, repo_id, *, gated=None, private=None, visibility=None, token=None,
                                     repo_type=None):
                hub.calls.append(("update_repo_settings", repo_id, repo_type, private))

            def upload_folder(self, *, repo_id, folder_path, path_in_repo=None, commit_message=None,
                              commit_description=None, token=None, repo_type=None, revision=None, create_pr=None,
                              parent_commit=None, allow_patterns=None, ignore_patterns=None, delete_patterns=None,
                              run_as_future=False):
                files = {p.relative_to(folder_path).as_posix(): p.read_text(encoding="utf-8")
                         for p in Path(folder_path).rglob("*") if p.is_file()}
                hub.uploads[repo_id] = files
                hub.calls.append(("upload_folder", repo_id, repo_type))

            def create_collection(self, title, *, namespace=None, description=None, private=False,
                                  resource_group_id=None, exists_ok=False, token=None):
                assert description is None or len(description) <= 150  # Hub limit per the docstring
                hub.calls.append(("create_collection", title, namespace, private, exists_ok))
                return Collection()

            def update_collection_metadata(self, collection_slug, *, title=None, description=None, position=None,
                                           private=None, theme=None, token=None):
                hub.calls.append(("update_collection_metadata", collection_slug, private))

            def add_collection_item(self, collection_slug, item_id, item_type, *, note=None, exists_ok=False,
                                    token=None):
                assert item_type in ("model", "dataset", "space", "paper", "collection", "bucket")
                hub.calls.append(("add_collection_item", item_id, item_type, exists_ok))

            def list_models(self, *, filter=None, author=None, search=None, limit=None, **kwargs):
                hub.calls.append(("list_models", author, search))
                return [types.SimpleNamespace(id=m) for m in hub.models if search in m and m.startswith(author + "/")]

        self.module = types.ModuleType("huggingface_hub")
        self.module.HfApi = HfApi


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub(models=["org/dhanground-A2-s0-abc1234", "org/dhanground-A3-s0-abc1234", "org/other-model",
                           "elsewhere/dhanground-A1-s0-abc1234"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake.module)
    return fake


def test_dry_run_prints_plan_and_uploads_nothing(repo, hub, capsys):
    assert publish_hf.main(["--org", "org", "--root", str(repo)]) == 0
    out = capsys.readouterr().out
    for name in ("org/dhanground-rbi-corpus", "org/dhanground-hinglish-qa", "org/dhanground-dpo-pairs"):
        assert f"dataset {name} (private)" in out
    assert "corpus/faq-2.jsonl" in out and "2 rows" in out
    assert "frozen_test/test.sha256" in out and "results/A2_s0_test.json" in out
    assert "org/dhanground-A3-s0-abc1234" in out and "org/dhanground-A1-s1-abc1234" in out
    assert "dry run" in out
    assert hub.calls == []


@pytest.mark.parametrize("breaker, message", [
    (lambda r: (r / "data/arms/A3_pairs.jsonl").unlink(), "missing data/arms/A3_pairs.jsonl"),
    (lambda r: [p.unlink() for p in (r / "corpus").glob("*.jsonl")], "missing corpus/*.jsonl"),
    (lambda r: [p.unlink() for p in (r / "evals/results").iterdir()], "missing evals/results"),
    (lambda r: (r / "data/test/test.jsonl").write_text("{}\n"), "does not match test.sha256"),
    (lambda r: (r / "evals/test_set_card.md").write_text(make_test_card.PLACEHOLDER_MARK), "placeholder"),
    (lambda r: (r / "data/curated/dev.jsonl").write_text(""), "data/curated/dev.jsonl is empty"),
])
def test_refuses_bad_inputs_even_with_push(repo, hub, capsys, breaker, message):
    breaker(repo)
    assert publish_hf.main(["--org", "org", "--root", str(repo), "--push", "--public"]) == 2
    assert message in capsys.readouterr().err
    assert hub.calls == []


def test_refuses_stale_test_card(repo, hub, capsys):
    rows = (repo / "data/test/test.jsonl").read_text(encoding="utf-8") + json.dumps(qa_row(4)) + "\n"
    (repo / "data/test/test.jsonl").write_text(rows, encoding="utf-8")
    digest = hashlib.sha256((repo / "data/test/test.jsonl").read_bytes()).hexdigest()
    (repo / "data/test/test.sha256").write_text(digest + "\n")
    assert publish_hf.main(["--org", "org", "--root", str(repo)]) == 2
    assert "stale" in capsys.readouterr().err


def test_push_private_uploads_datasets_cards_and_collection(repo, hub):
    assert publish_hf.main(["--org", "org", "--root", str(repo), "--push"]) == 0
    creates = [c for c in hub.calls if c[0] == "create_repo"]
    assert creates == [("create_repo", f"org/{n}", "dataset", True, True)
                       for n in ("dhanground-rbi-corpus", "dhanground-hinglish-qa", "dhanground-dpo-pairs")]
    assert not any(c[0] in ("update_repo_settings", "update_collection_metadata") for c in hub.calls)

    corpus = hub.uploads["org/dhanground-rbi-corpus"]
    assert set(corpus) == {"README.md", "corpus/rbi-1.jsonl", "corpus/faq-2.jsonl"}
    qa = hub.uploads["org/dhanground-hinglish-qa"]
    assert set(qa) == {"README.md", "data/train.jsonl", "data/dev.jsonl", "data/test.jsonl", "data/gate_log.jsonl",
                       "frozen_test/test.jsonl", "frozen_test/test.sha256", "test_set_card.md",
                       "results/A2_s0_test.json", "results/summary_test.md"}
    assert qa["frozen_test/test.jsonl"] == (repo / "data/test/test.jsonl").read_text(encoding="utf-8")
    assert set(hub.uploads["org/dhanground-dpo-pairs"]) == {"README.md", "data/A3_pairs.jsonl"}

    assert ("create_collection", "DhanGround", "org", True, True) in hub.calls
    items = [(c[1], c[2]) for c in hub.calls if c[0] == "add_collection_item"]
    assert items == [("org/dhanground-rbi-corpus", "dataset"), ("org/dhanground-hinglish-qa", "dataset"),
                     ("org/dhanground-dpo-pairs", "dataset"),
                     ("org/dhanground-A2-s0-abc1234", "model"), ("org/dhanground-A3-s0-abc1234", "model")]


def test_push_public_flips_existing_repos_and_collection(repo, hub):
    assert publish_hf.main(["--org", "org", "--root", str(repo), "--push", "--public"]) == 0
    assert all(c[3] is False for c in hub.calls if c[0] == "create_repo")
    flipped = {(c[1], c[2]) for c in hub.calls if c[0] == "update_repo_settings" and c[3] is False}
    assert flipped == {("org/dhanground-rbi-corpus", "dataset"), ("org/dhanground-hinglish-qa", "dataset"),
                       ("org/dhanground-dpo-pairs", "dataset"), ("org/dhanground-A2-s0-abc1234", "model"),
                       ("org/dhanground-A3-s0-abc1234", "model")}
    assert ("create_collection", "DhanGround", "org", False, True) in hub.calls
    assert any(c[0] == "update_collection_metadata" and c[2] is False for c in hub.calls)


def test_dataset_cards_have_fields_provenance_licence_and_snapshot(repo, hub):
    publish_hf.main(["--org", "org", "--root", str(repo), "--push"])
    for repo_id, files in hub.uploads.items():
        card = files["README.md"]
        assert card.startswith("---\n") and "license: other" in card
        for section in ("## Files", "## Fields", "## Provenance", "## Licence"):
            assert section in card, (repo_id, section)
        assert "Corpus snapshot date: 2026-11-10" in card
        assert "non-commercial" in card and "rbi.org.in" in card
    qa = hub.uploads["org/dhanground-hinglish-qa"]["README.md"]
    assert "- `answerable`:" in qa and "- `gate`:" in qa and "| `data/train.jsonl` | 2 |" in qa
    assert "- `reject_reason`:" in hub.uploads["org/dhanground-dpo-pairs"]["README.md"]


def test_test_card_counts_and_hash(repo):
    card = (repo / "evals/test_set_card.md").read_text(encoding="utf-8")
    digest = (repo / "data/test/test.sha256").read_text().strip()
    assert "The frozen test set has 3 rows." in card
    assert f"`{digest}`. It matches test.sha256" in card
    assert "| answerable | 1 | 33% |" in card and "| adversarial | 1 | 33% |" in card
    assert "| hi-Deva | 1 | 33% |" in card and "| en | 1 | 33% |" in card
    assert "| upi | 1 | 33% |" in card and "| atm | 1 | 33% |" in card and "| injection | 1 | 33% |" in card
    assert "They kept 2 rows (keep=y) and removed 1 rows (keep=n)." in card
    assert make_test_card.PLACEHOLDER_MARK not in card


def test_test_card_warns_on_hash_mismatch(repo):
    (repo / "data/test/test.sha256").write_text("0" * 64 + "\n")
    card = make_test_card.build_card(repo / "data/test/test.jsonl", repo / "data/test/review.csv", repo / "corpus")
    assert "WARNING" in card and "Do not report results on it." in card


def test_committed_test_card_is_the_placeholder_until_freeze():
    text = (Path(__file__).resolve().parent.parent / "evals/test_set_card.md").read_text(encoding="utf-8")
    assert text.startswith(make_test_card.PLACEHOLDER_MARK)


def test_upstream_issues_doc_is_an_empty_template():
    text = (Path(__file__).resolve().parent.parent / "docs/upstream_issues.md").read_text(encoding="utf-8")
    assert "| Tool | Version | Issue | Repro | Status | Link |" in text
    rows = [l for l in text.splitlines() if l.startswith("|")][2:]
    assert all(set(c.strip() for c in r.strip("|").split("|")) == {""} for r in rows), "no invented entries"
