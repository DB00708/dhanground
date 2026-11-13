"""The stage notebooks are valid nbformat 4 JSON with no outputs, and every call in them matches the real code.

Each `!{PY} -m module ...` line is filled with sample values and parsed by that module's own argparse parser,
so a misspelled flag, a wrong choice or a missing positional fails here. Direct Python calls to repo functions
are bound against the real signatures.
"""
import argparse
import ast
import importlib
import inspect
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ["01_ingest", "02_generate", "03_curate", "04_train_sft", "05_mine_and_dpo", "06_eval", "07_roi"]
REPO_PACKAGES = {p.name for p in ROOT.iterdir() if (p / "__init__.py").is_file()}

CLI_RE = re.compile(r"-m\s+([A-Za-z_][\w.]*)")
FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b", re.M)
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w., ]*)$", re.M)
SHELL_RE = re.compile(r"^\s*!\{PY\}\s+-m\s+([\w.]+)(.*)$")
VAR_RE = re.compile(r"\{([^{}]+)\}")

# Sample values for the variables the notebooks put into CLI lines. A new variable must be added here.
SAMPLE = {
    "CORPUS": "corpus/", "SPLITS": "data/splits.json", "INDEX": "index/rbi-2026-11-10",
    "arm": "A1", "seed": "0", "BACKEND": "aligntune", "HUB_ARG": "--hub-org my-org",
    "A2": "checkpoints/A2-s0-abc1234", "DEV_ADAPTER": "checkpoints/A2-s0-abc1234",
    "adapter_arg": "--adapter checkpoints/A1-s0-abc1234", "ARM": "A3", "P_IN": "2.5", "P_OUT": "10",
}


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


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def cli_parser(module: str, monkeypatch) -> argparse.ArgumentParser:
    """The ArgumentParser that `python -m module` builds, captured when main() calls parse_args."""
    def capture(self, *a, **k):
        raise _Captured(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    mod = importlib.import_module(module)
    with pytest.raises(_Captured) as exc:
        mod.main()
    monkeypatch.undo()
    parser = exc.value.parser

    def fail(message):
        raise ValueError(message)

    parser.error = fail
    return parser


def shell_commands(name: str) -> list[tuple[str, list[str]]]:
    out = []
    for src in code_sources(load(name)):
        for line in src.splitlines():
            m = SHELL_RE.match(line)
            if not m:
                continue
            module, rest = m.groups()
            unknown = [v for v in VAR_RE.findall(rest) if v not in SAMPLE]
            assert not unknown, f"{name}: add sample values for {unknown} to SAMPLE"
            out.append((module, shlex.split(VAR_RE.sub(lambda v: SAMPLE[v.group(1)], rest))))
    return out


def check_cli(module: str, argv: list[str], monkeypatch) -> None:
    cli_parser(module, monkeypatch).parse_args(argv)


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


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_cli_lines_parse_with_the_real_argparse(name, monkeypatch):
    for module, argv in shell_commands(name):
        check_cli(module, argv, monkeypatch)


def test_cli_check_catches_a_misspelled_flag(monkeypatch):
    check_cli("evals.run_eval", ["--arm", "A2", "--seed", "0", "--split", "test"], monkeypatch)
    with pytest.raises(ValueError, match="unrecognized"):
        check_cli("evals.run_eval", ["--arm", "A2", "--seeds", "0"], monkeypatch)
    with pytest.raises(ValueError, match="invalid choice"):
        check_cli("adapters.aligntune_sft", ["--arm", "A3"], monkeypatch)
    with pytest.raises(ValueError, match="required"):
        check_cli("retrieval.build_index", ["corpus/", "index/"], monkeypatch)


def test_every_runbook_stage_is_in_a_notebook():
    used = {m for name in NOTEBOOKS for m, _ in shell_commands(name)}
    used |= {m for name in NOTEBOOKS for src in code_sources(load(name)) for m in FROM_RE.findall(src)}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    runbook = readme[readme.index("**Corpus and split.**"):readme.index("**Demo.**")]
    stages = set(re.findall(r"python -m ([\w.]+)", runbook))
    assert stages - used == set(), f"runbook stages missing from notebooks: {sorted(stages - used)}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_python_calls_match_signatures(name):
    """`from repo.module import f` then `f(...)` must bind to f's real signature."""
    for src in code_sources(load(name)):
        # `!` shell lines become `pass` at the same indent so loops around them still parse
        lines = [l[: len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith("!") else l
                 for l in src.splitlines()]
        tree = ast.parse("\n".join(lines))
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in REPO_PACKAGES:
                mod = importlib.import_module(node.module)
                funcs |= {a.asname or a.name: getattr(mod, a.name) for a in node.names}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                sig = inspect.signature(funcs[node.func.id])
                sig.bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})


def test_eval_notebook_keeps_the_once_per_test_guard():
    src = "\n".join(code_sources(load("06_eval")))
    assert "--force" not in src
    test_lines = [l for l in src.splitlines() if "run_eval" in l and "--split test" in l]
    assert len(test_lines) == 1
    assert "test_usage.json" in src and "_exit_code" in src
    # adapters are resolved for all runs before the first test run
    assert src.index("PLAN = [") < src.index(test_lines[0].strip())
