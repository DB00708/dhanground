"""Build an AnswerEngine from configs: base model + whichever arm adapters exist."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def adapter_paths() -> dict[str, str]:
    """ADAPTER_A1=... env vars (local path or HF repo id) override configs/serving.yaml."""
    cfg_path = Path("configs/serving.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    paths = {arm: p for arm, p in (cfg.get("adapters") or {}).items() if p}
    for arm in ("A1", "A2", "A3"):
        if os.environ.get(f"ADAPTER_{arm}"):
            paths[arm] = os.environ[f"ADAPTER_{arm}"]
    return paths


def build_engine(index_dir: str | None = None, arms: list[str] | None = None):
    from serving.engine import AnswerEngine
    from retrieval.search import Retriever

    arms_cfg = yaml.safe_load(Path("configs/arms.yaml").read_text(encoding="utf-8"))
    ret_cfg = yaml.safe_load(Path("configs/retrieval.yaml").read_text(encoding="utf-8"))
    paths = adapter_paths()
    if arms is not None:
        paths = {a: p for a, p in paths.items() if a in arms}
    base = os.environ.get("BASE_MODEL", arms_cfg["base_model"])
    retriever = None
    index_dir = index_dir or os.environ.get("INDEX_DIR")
    if index_dir:
        retriever = Retriever(Path(index_dir), ret_cfg["hybrid"]["bm25_weight"], ret_cfg["hybrid"]["embedding_weight"])
    return AnswerEngine(_generator(base, paths), retriever, tau=ret_cfg.get("tau"), k=ret_cfg["k"])


def _generator(base: str, paths: dict[str, str]):
    """DHANGROUND_GENERATOR=module:Class swaps in a stub generator (e2e tests; no GPU or weights)."""
    from serving.engine import HFGenerator

    spec = os.environ.get("DHANGROUND_GENERATOR")
    if not spec:
        return HFGenerator(base, paths)
    import importlib

    module, _, cls = spec.partition(":")
    return getattr(importlib.import_module(module), cls)(base, paths)
