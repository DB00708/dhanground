"""Component 1: corpus ingester. RBI URL allowlist -> corpus/{doc_id}.jsonl, one row per paragraph.

Usage: python -m pipeline.ingest configs/rbi_sources.yaml corpus/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import requests
import trafilatura
import yaml

from core import state

ALLOWED_HOSTS = {"rbi.org.in", "www.rbi.org.in", "rbidocs.rbi.org.in", "m.rbi.org.in"}
MIN_PARA_CHARS = 40
MAX_CHUNK_WORDS = 100
MIN_CHUNK_WORDS = 25
USER_AGENT = "dhanground-research/0.1 (non-commercial research)"


def check_allowed(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"not an allowlisted RBI URL: {url}")


def fetch(url: str, retries: int = 3) -> requests.Response:
    check_allowed(url)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_ROW_ID_RE = re.compile(r"^(\d+|[a-z]|[ivx]+)[.)]?$", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.;:?])\s+(?=[A-Z0-9‘“(])")
_NUMBERED_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3})*\.(?:\s|$)|^\d{1,3}(?:\.\d{1,3})+(?:\s|$)")


def chunk_prose(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Pack sentences into chunks of at most max_words; a numbered item ('4. Who is...') starts a new chunk."""
    chunks, current, count = [], [], 0
    for sentence in _SENTENCE_RE.split(normalize(text)):
        n = len(sentence.split())
        numbered = _NUMBERED_RE.match(sentence) is not None
        if current and (count + n > max_words or (numbered and count >= MIN_CHUNK_WORDS)):
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str) -> list[str]:
    """Prose -> sentence-packed chunks. Table rows -> one chunk each, prefixed with their section heading.

    Merged-cell rows with an empty rule column are kept as-is; copying the rule down from the
    row above could attach the wrong timeline to an incident.
    """
    chunks: list[str] = []
    section = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("|"):
            chunks.extend(chunk_prose(line))
            continue
        cells = [normalize(c) for c in line.strip("|").split("|")]
        nonempty = [c for c in cells if c]
        if not nonempty:
            continue
        if len(nonempty) == 1:
            chunks.extend(chunk_prose(nonempty[0]))
            continue
        first = cells[0]
        if not _ROW_ID_RE.match(first):
            continue  # column header row
        rest = [c for c in cells[1:] if c]
        if first.isdigit() and len(rest) == 1:
            section = rest[0]
            continue
        body = " | ".join(rest)
        chunks.append(f"{section}: {body}" if section else body)
    seen: set[str] = set()
    return [c for c in chunks if len(c) >= MIN_PARA_CHARS and not (c in seen or seen.add(c))]


def extract_html(html: str) -> list[str]:
    return chunk_text(trafilatura.extract(html, include_tables=True, favor_recall=True) or "")


_PAGE_NUM_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)


def extract_pdf(data: bytes) -> list[str]:
    """Text blocks in reading order; drops page numbers and lines repeated on many pages (headers/footers)."""
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        pages = [
            [normalize(b[4]) for b in page.get_text("blocks", sort=True) if b[6] == 0]
            for page in pdf
        ]
    counts: dict[str, int] = {}
    for blocks in pages:
        for b in set(blocks):
            counts[b] = counts.get(b, 0) + 1
    repeated = {b for b, n in counts.items() if len(pages) >= 4 and n >= 0.3 * len(pages)}
    kept = [b for blocks in pages for b in blocks if b and b not in repeated and not _PAGE_NUM_RE.match(b)]
    return chunk_text(" ".join(kept))


class BlockedError(Exception):
    pass


def extract_paragraphs(content: bytes, text: str | None = None) -> list[str]:
    if content[:5] == b"%PDF-":
        return extract_pdf(content)
    text = text if text is not None else content.decode("utf-8", errors="replace")
    if "testing whether you are a human" in text:
        raise BlockedError("RBI returned a human-verification page; download this file by hand")
    return extract_html(text)


def load_document(doc: dict, root: Path) -> list[str]:
    """A local 'file' wins (PDFs on rbidocs.rbi.org.in are behind a CAPTCHA for scripts); else fetch 'url'."""
    check_allowed(doc["url"])
    if doc.get("file"):
        path = root / doc["file"]
        if not path.exists():
            raise BlockedError(f"missing {doc['file']}; download {doc['url']} in a browser and save it there")
        return extract_paragraphs(path.read_bytes())
    resp = fetch(doc["url"])
    return extract_paragraphs(resp.content, resp.text)


def build_rows(doc: dict, paragraphs: list[str]) -> list[dict]:
    rows = []
    for i, text in enumerate(paragraphs, start=1):
        para_id = f"p{i}"
        rows.append({
            "pid": f"{doc['doc_id']}-{para_id}",
            "doc_id": doc["doc_id"],
            "para_id": para_id,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "title": doc["title"],
            "url": doc["url"],
            "date": doc["date"],
            "topics": doc.get("topics", []),
        })
    return rows


def ingest(sources_path: Path, out_dir: Path) -> int:
    docs = yaml.safe_load(sources_path.read_text(encoding="utf-8"))["documents"]
    root = sources_path.resolve().parent.parent
    local = [root / d["file"] for d in docs if d.get("file") and (root / d["file"]).exists()]
    input_hash = state.hash_inputs([sources_path, *local])
    if state.is_fresh(out_dir, input_hash):
        print(f"{out_dir} is up to date; skipping")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    total, failed = 0, []
    for doc in docs:
        try:
            rows = build_rows(doc, load_document(doc, root))
        except (requests.RequestException, ValueError, BlockedError) as e:
            failed.append(doc["doc_id"])
            print(f"FAILED {doc['doc_id']}: {e}", file=sys.stderr)
            continue
        if not rows:
            failed.append(doc["doc_id"])
            print(f"FAILED {doc['doc_id']}: no paragraphs extracted", file=sys.stderr)
            continue
        with (out_dir / f"{doc['doc_id']}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{doc['doc_id']}: {len(rows)} paragraphs")
    if failed:
        print(f"{len(failed)} document(s) failed; manifest not written so the stage reruns", file=sys.stderr)
        return 1
    state.write_manifest(out_dir, "INGESTED", input_hash, documents=len(docs), paragraphs=total)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    sys.exit(ingest(args.sources, args.out_dir))


if __name__ == "__main__":
    main()
