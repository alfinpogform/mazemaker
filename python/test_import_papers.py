#!/usr/bin/env python3
"""test_import_papers.py — hermetic tests for import_papers.py.

No live network calls or PDF library required: arXiv parsing is tested
against a canned Atom response, network/PDF-dependent paths are exercised
only through their pure error/fallback behavior.

Run: python3 python/test_import_papers.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0

SAMPLE_ARXIV_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>
   Attention Is All You Need
    </title>
    <summary>
  The dominant sequence transduction models are based on complex
  recurrent or convolutional neural networks.
    </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""

EMPTY_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
"""


def _testcase(name):
    def deco(fn):
        def wrapper():
            global PASS, FAIL
            try:
                fn()
                print(f"  PASS  {name}")
                PASS += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                FAIL += 1
        return wrapper
    return deco


@_testcase("extract_arxiv_id: bare id")
def t1():
    from import_papers import extract_arxiv_id
    assert extract_arxiv_id("1706.03762") == "1706.03762"


@_testcase("extract_arxiv_id: abs URL")
def t2():
    from import_papers import extract_arxiv_id
    assert extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"


@_testcase("extract_arxiv_id: pdf URL with version suffix")
def t3():
    from import_papers import extract_arxiv_id
    assert extract_arxiv_id("https://arxiv.org/pdf/1706.03762v5.pdf") == "1706.03762"


@_testcase("extract_arxiv_id: non-arxiv reference returns None")
def t4():
    from import_papers import extract_arxiv_id
    assert extract_arxiv_id("https://example.com/some-article") is None
    assert extract_arxiv_id("not a paper reference at all") is None


@_testcase("parse_arxiv_entry: extracts title/abstract/authors, collapses whitespace")
def t5():
    from import_papers import parse_arxiv_entry
    meta = parse_arxiv_entry(SAMPLE_ARXIV_ATOM, "1706.03762")
    assert meta["title"] == "Attention Is All You Need"
    assert "recurrent or convolutional" in meta["abstract"]
    assert "\n" not in meta["abstract"]
    assert meta["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert meta["url"] == "https://arxiv.org/abs/1706.03762"


@_testcase("parse_arxiv_entry: raises on empty feed")
def t6():
    from import_papers import parse_arxiv_entry
    try:
        parse_arxiv_entry(EMPTY_ATOM, "0000.00000")
        assert False, "expected ValueError for empty feed"
    except ValueError:
        pass


@_testcase("extract_pdf_text: missing file / no PDF lib returns None, doesn't raise")
def t7():
    from import_papers import extract_pdf_text
    result = extract_pdf_text(Path("/nonexistent/path/paper.pdf"))
    assert result is None


@_testcase("ingest: bare URL without --title raises a clear error")
def t8():
    from import_papers import ingest
    try:
        ingest("https://example.com/some-article")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "title" in str(e)


@_testcase("ingest: bare URL with --title stores as a url-sourced reference")
def t9():
    from import_papers import ingest
    fields = ingest("https://example.com/some-article", title="Some Article", tags="misc")
    assert fields["title"] == "Some Article"
    assert fields["source"] == "url"
    assert fields["url"] == "https://example.com/some-article"
    assert fields["source_id"] == "https://example.com/some-article"
    assert fields["tags"] == "misc"


@_testcase("ingest: nonexistent .pdf path falls through to bare-URL handling")
def t10():
    from import_papers import ingest
    # Doesn't exist on disk, and isn't an arXiv reference either — needs --title.
    try:
        ingest("/nonexistent/paper.pdf")
        assert False, "expected ValueError"
    except ValueError:
        pass


@_testcase("ingest: existing PDF without a PDF library still stores file_path")
def t11():
    from import_papers import ingest
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 not a real pdf, just needs to exist on disk")
        path = f.name
    try:
        fields = ingest(path)
        assert fields["source"] == "file"
        assert fields["file_path"] == path
        assert fields["title"]  # falls back to the filename stem
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  Paper Import Tests")
    print("=" * 60)
    for t in [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]:
        t()
    print("=" * 60)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
