#!/usr/bin/env python3
"""test_paper_library.py — tests for the AI-papers/documents storage space.

Run: python3 python/test_paper_library.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = 0
FAIL = 0


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


def temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def cleanup(path):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except FileNotFoundError:
            pass


@_testcase("add + get roundtrip")
def t1():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            pid = lib.add(
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer"],
                abstract="The dominant sequence transduction models are based on RNNs.",
                source="arxiv", source_id="1706.03762",
                url="https://arxiv.org/abs/1706.03762",
                tags=["transformers", "nlp"],
            )
            assert pid > 0
            p = lib.get(pid)
            assert p["title"] == "Attention Is All You Need"
            assert p["authors"] == ["Vaswani", "Shazeer"]
            assert p["tags"] == ["transformers", "nlp"]
            assert p["source_id"] == "1706.03762"
    finally:
        cleanup(db)


@_testcase("add upserts on duplicate source_id")
def t2():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            id1 = lib.add(title="Draft Title", source_id="arxiv:1234", tags=["draft"])
            id2 = lib.add(title="Final Title", source_id="arxiv:1234", tags=["final"])
            assert id1 == id2, "same source_id should update in place, not duplicate"
            assert lib.stats()["papers"] == 1
            p = lib.get(id1)
            assert p["title"] == "Final Title"
            assert p["tags"] == ["final"]
    finally:
        cleanup(db)


@_testcase("list filters by tag and source")
def t3():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            lib.add(title="Paper A", source="arxiv", tags=["nlp", "transformers"])
            lib.add(title="Paper B", source="arxiv", tags=["vision"])
            lib.add(title="Paper C", source="note", tags=["nlp"])
            assert len(lib.list()) == 3
            assert len(lib.list(tag="nlp")) == 2
            assert len(lib.list(source="arxiv")) == 2
            assert len(lib.list(tag="nlp", source="note")) == 1
    finally:
        cleanup(db)


@_testcase("delete removes the paper")
def t4():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            pid = lib.add(title="Ephemeral Paper")
            assert lib.delete(pid) is True
            assert lib.get(pid) is None
            assert lib.delete(pid) is False  # already gone
    finally:
        cleanup(db)


@_testcase("search finds relevant papers by keyword")
def t5():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            lib.add(title="Deep Reinforcement Learning for Robotics",
                    abstract="We study policy gradient methods for robotic control.")
            lib.add(title="Sourdough Bread Baking Guide",
                    abstract="A guide to fermentation and crust texture.")
            results = lib.search("reinforcement learning robotics", k=5)
            assert len(results) >= 1
            assert results[0]["title"] == "Deep Reinforcement Learning for Robotics"
    finally:
        cleanup(db)


@_testcase("search on empty library returns no results")
def t6():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            assert lib.search("anything") == []
            assert lib.search("") == []
    finally:
        cleanup(db)


@_testcase("stats report counts by source and embedded fraction")
def t7():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            lib.add(title="A", source="arxiv")
            lib.add(title="B", source="arxiv")
            lib.add(title="C", source="url", embed=False)
            s = lib.stats()
            assert s["papers"] == 3
            assert s["by_source"]["arxiv"] == 2
            assert s["embedded"] == 2  # C was added with embed=False
    finally:
        cleanup(db)


@_testcase("persistence across reopen")
def t8():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        lib = PaperLibrary(db, embedding_backend="hash")
        lib.add(title="Persistent Paper", source_id="doi:10.1/xyz")
        lib.close()

        lib2 = PaperLibrary(db, embedding_backend="hash")
        results = lib2.list()
        assert len(results) == 1
        assert results[0]["title"] == "Persistent Paper"
        lib2.close()
    finally:
        cleanup(db)


@_testcase("title is required")
def t9():
    from paper_library import PaperLibrary
    db = temp_db()
    try:
        with PaperLibrary(db, embedding_backend="hash") as lib:
            try:
                lib.add(title="")
                assert False, "expected ValueError for empty title"
            except ValueError:
                pass
    finally:
        cleanup(db)


if __name__ == "__main__":
    print("=" * 60)
    print("  Paper Library Tests")
    print("=" * 60)
    for t in [t1, t2, t3, t4, t5, t6, t7, t8, t9]:
        t()
    print("=" * 60)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
