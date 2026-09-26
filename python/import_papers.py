#!/usr/bin/env python3
"""import_papers.py — ingest AI papers/documents into the paper library.

Supports three kinds of reference:
  - arXiv id or arxiv.org URL  → fetches title/authors/abstract from arXiv's
                                  public Atom API, dedupes on the arXiv id
  - local PDF path             → extracts text (via pypdf, if installed) as
                                  full_text; stored with just the file path
                                  if no PDF library is available
  - a bare URL + --title       → stored as a reference with no fetched
                                  metadata (title is required since there's
                                  nothing to look up)

Usage:
    python3 python/import_papers.py 1706.03762
    python3 python/import_papers.py https://arxiv.org/abs/1706.03762 --tags nlp,transformers
    python3 python/import_papers.py ~/papers/some-paper.pdf --tags rl
    python3 python/import_papers.py https://example.com/article --title "Some Article"
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
ARXIV_BARE_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?$")
ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def extract_arxiv_id(ref: str) -> Optional[str]:
    """Pull a bare arXiv id (e.g. '1706.03762') out of a bare id or an
    arxiv.org URL. Deliberately does NOT match a YYYY.NNNNN-shaped number
    appearing anywhere in an arbitrary non-arxiv.org URL — that would force
    a live arXiv fetch (and can fail outright) for references that were
    never arXiv papers to begin with.
    """
    ref = ref.strip()
    m = ARXIV_BARE_ID_RE.match(ref)
    if m:
        return m.group(1)
    host = urllib.parse.urlparse(ref).netloc.lower()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        m = ARXIV_ID_RE.search(ref)
        if m:
            return m.group(1)
    return None


def parse_arxiv_entry(xml_bytes: bytes, arxiv_id: str) -> dict:
    """Parse one <entry> out of an arXiv Atom API response."""
    root = ET.fromstring(xml_bytes)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise ValueError(f"arXiv API returned no entry for id {arxiv_id}")
    title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
    authors = [
        (a.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
        for a in entry.findall("atom:author", ATOM_NS)
    ]
    if not title:
        raise ValueError(f"arXiv entry for id {arxiv_id} has no title — malformed response?")
    return {
        # Atom titles/summaries wrap with embedded newlines; collapse whitespace.
        "title": " ".join(title.split()),
        "abstract": " ".join(summary.split()),
        "authors": [a for a in authors if a],
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def fetch_arxiv_metadata(arxiv_id: str, timeout: float = 15.0) -> dict:
    """Fetch title/authors/abstract for an arXiv id via the public Atom API."""
    url = ARXIV_API.format(id=arxiv_id)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        xml_bytes = resp.read()
    return parse_arxiv_entry(xml_bytes, arxiv_id)


def extract_pdf_text(path: Path, max_chars: int = 50_000) -> Optional[str]:
    """Best-effort text extraction. Returns None if no PDF library is available
    or extraction fails — the paper is still stored with just its file path."""
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
    except ImportError:
        return None

    try:
        reader = PdfReader(str(path))
        chunks = []
        total = 0
        for page in reader.pages:
            chunk = page.extract_text() or ""
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
        text = "\n".join(chunks).strip()
        return text[:max_chars] or None
    except Exception:
        return None


def ingest(ref: str, *, title: Optional[str] = None, tags: Optional[str] = None,
           source: Optional[str] = None) -> dict:
    """Resolve one reference (arXiv id/URL, PDF path, or bare URL) into the
    field dict PaperLibrary.add() expects, without touching any store —
    kept separate from storage so this is unit-testable without a live
    network connection or a PDF library installed.
    """
    path = Path(ref).expanduser()
    if path.suffix.lower() == ".pdf" and path.exists():
        return {
            "title": title or path.stem,
            "source": source or "file",
            "file_path": str(path),
            "full_text": extract_pdf_text(path),
            "tags": tags,
        }

    arxiv_id = extract_arxiv_id(ref)
    if arxiv_id:
        meta = fetch_arxiv_metadata(arxiv_id)
        return {
            "title": title or meta["title"],
            "authors": meta["authors"],
            "abstract": meta["abstract"],
            "source": source or "arxiv",
            "source_id": arxiv_id,
            "url": meta["url"],
            "tags": tags,
        }

    if not title:
        raise ValueError(
            f"'{ref}' isn't an arXiv id/URL or an existing PDF path — "
            "pass --title to store it as a bare URL reference"
        )
    return {
        "title": title,
        "source": source or "url",
        "source_id": ref,
        "url": ref,
        "tags": tags,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Import an AI paper/document into the paper library"
    )
    ap.add_argument("ref", help="arXiv id, arxiv.org URL, local PDF path, or a bare URL (with --title)")
    ap.add_argument("--title", help="Override or supply the title (required for bare URLs)")
    ap.add_argument("--tags", help="Comma-separated tags")
    ap.add_argument("--source", help="Override the inferred source label")
    ap.add_argument("--db", help="Path to the paper library DB (default: ~/.mazemaker/papers/library.db)")
    args = ap.parse_args()

    try:
        fields = ingest(args.ref, title=args.title, tags=args.tags, source=args.source)
    except (ValueError, urllib.error.URLError, ET.ParseError) as e:
        print(f"Import failed: {e}", file=sys.stderr)
        sys.exit(1)

    from paper_library import PaperLibrary
    lib = PaperLibrary(db_path=args.db) if args.db else PaperLibrary()
    with lib:
        pid = lib.add(**fields)
        print(f"Stored paper id={pid}: {fields['title']}")


if __name__ == "__main__":
    main()
