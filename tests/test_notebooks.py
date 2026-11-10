"""The stage notebooks are valid nbformat 4 JSON, carry no outputs, and only reference modules that exist."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ["01_ingest", "02_generate", "03_curate", "04_train_sft", "05_mine_and_dpo", "06_eval", "07_roi"]
REPO_PACKAGES = {p.name for p in ROOT.iterdir() if (p / "__init__.py").is_file()}

CLI_RE = re.compile(r"-m\s+([A-Za-z_][\w.]*)")
FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b", re.M)
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w., ]*)$", re.M)


def load(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))


def code_sources(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def resolves(module: str) -> bool:
    parts = module.split(".")
    if parts[0] not in REPO_PACKAGES:
        return parts[0] in sys.stdlib_module_names
    path = ROOT.joinpath(*parts)
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_nbformat(name):
    nb = load(name)
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    assert nb["cells"][0]["cell_type"] == "markdown" and "".join(nb["cells"][0]["source"]).startswith("# ")
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code") and isinstance(c["source"], list)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_outputs(name):
    for c in load(name)["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_modules_resolve(name):
    referenced = set()
    for src in code_sources(load(name)):
        referenced |= set(CLI_RE.findall(src)) | set(FROM_RE.findall(src))
        for group in IMPORT_RE.findall(src):
            referenced |= {m.strip().split(" as ")[0] for m in group.split(",")}
    assert any(m.split(".")[0] in REPO_PACKAGES for m in referenced), "notebook calls no repo module"
    missing = sorted(m for m in referenced if not resolves(m))
    assert not missing, f"unresolved modules: {missing}"
