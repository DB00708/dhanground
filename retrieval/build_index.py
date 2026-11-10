"""Build the retrieval index from corpus/*.jsonl.

Usage: python -m retrieval.build_index corpus/ index/ --version rbi-2026-11-10 [--embedding-model intfloat/multilingual-e5-small]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core import state


def build(corpus_dir: Path, out_root: Path, version: str, embedding_model: str | None) -> Path:
    files = sorted(corpus_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no corpus files in {corpus_dir}")
    out = out_root / version
    input_hash = state.hash_inputs(files, {"embedding_model": embedding_model})
    if state.is_fresh(out, input_hash):
        print(f"{out} is up to date")
        return out
    passages = [json.loads(l) for f in files for l in f.open(encoding="utf-8")]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"index_version": version, "embedding_model": embedding_model, "passages": len(passages)}
    if embedding_model:
        from sentence_transformers import SentenceTransformer

        is_e5 = "e5" in embedding_model.lower()
        meta["query_prefix"] = "query: " if is_e5 else ""
        enc = SentenceTransformer(embedding_model)
        texts = [("passage: " if is_e5 else "") + p["text"] for p in passages]
        emb = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        np.save(out / "embeddings.npy", np.asarray(emb, dtype=np.float32))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    state.write_manifest(out, "INGESTED", input_hash, **meta)
    print(f"indexed {len(passages)} passages -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--version", required=True)
    ap.add_argument("--embedding-model", default=None)
    a = ap.parse_args()
    build(a.corpus_dir, a.out_root, a.version, a.embedding_model)


if __name__ == "__main__":
    main()
