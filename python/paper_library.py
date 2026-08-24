"""paper_library.py — a dedicated storage space for AI papers and documents.

A small, self-contained SQLite store for research papers and reference docs
(arXiv papers, PDFs, web articles, notes) — separate from the general-purpose
memory store in memory_client.py, since papers have their own shape (title,
authors, abstract, a stable external id, tags) and don't need the graph/
connection/dream machinery that memories do.

Reuses embed_provider.EmbeddingProvider for semantic search so it shares the
same offline-first backend chain (shared server > sentence-transformers >
fastembed > tfidf > hash) as the rest of the codebase, but the embedder is
only loaded lazily — plain CRUD/browsing works with zero ML dependencies.

Usage:
    from paper_library import PaperLibrary

    lib = PaperLibrary()
    pid = lib.add(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="The dominant sequence transduction models...",
        source="arxiv", source_id="1706.03762",
        url="https://arxiv.org/abs/1706.03762",
        tags=["transformers", "nlp"],
    )
    lib.search("self-attention mechanism", k=5)
    lib.list(tag="nlp")
    lib.close()
"""
from __future__ import annotations

import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = Path.home() / ".mazemaker" / "papers" / "library.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    authors     TEXT,               -- comma-separated
    abstract    TEXT,
    source      TEXT,               -- e.g. 'arxiv', 'url', 'file', 'note'
    source_id   TEXT,               -- stable external id (arXiv id, DOI, URL) — dedupe key
    url         TEXT,
    file_path   TEXT,               -- local path to the stored PDF/doc, if any
    full_text   TEXT,               -- extracted text, used for FTS + embedding when present
    tags        TEXT,               -- comma-separated
    embedding   BLOB,               -- packed float32 vector, or NULL until embedded
    added_at    REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_source_id
    ON papers(source_id) WHERE source_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_added_at ON papers(added_at);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, authors, abstract, full_text, tags,
    content='papers',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, authors, abstract, full_text, tags)
    VALUES (new.id, new.title, new.authors, new.abstract, new.full_text, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, authors, abstract, full_text, tags)
    VALUES('delete', old.id, old.title, old.authors, old.abstract, old.full_text, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, authors, abstract, full_text, tags)
    VALUES('delete', old.id, old.title, old.authors, old.abstract, old.full_text, old.tags);
    INSERT INTO papers_fts(rowid, title, authors, abstract, full_text, tags)
    VALUES (new.id, new.title, new.authors, new.abstract, new.full_text, new.tags);
END;
"""


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: Optional[bytes]) -> Optional[list[float]]:
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 1e-10 else 0.0


def _norm_list(value) -> Optional[str]:
    """Accept a list/tuple or a pre-joined string; store as comma-separated text."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip() or None


class PaperLibrary:
    def __init__(self, db_path: "str | Path" = DEFAULT_DB_PATH, embedding_backend: str = "auto"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._embedding_backend = embedding_backend
        self._embedder = None  # lazy — plain CRUD shouldn't need ML deps
        self._lock = threading.Lock()

        self.conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(SCHEMA)
        self._fts_available = self._ensure_fts()
        self.conn.commit()

    # -- setup -----------------------------------------------------------

    def _ensure_fts(self) -> bool:
        try:
            self.conn.executescript(FTS_SCHEMA)
            paper_count = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            fts_count = self.conn.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0]
            if paper_count and fts_count == 0:
                self.conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
            return True
        except sqlite3.OperationalError:
            return False  # FTS5 unavailable in this SQLite build — search() falls back to LIKE

    def _get_embedder(self):
        if self._embedder is None:
            from embed_provider import EmbeddingProvider
            self._embedder = EmbeddingProvider(backend=self._embedding_backend)
        return self._embedder

    # -- CRUD --------------------------------------------------------------

    def add(
        self,
        title: str,
        *,
        authors=None,
        abstract: Optional[str] = None,
        source: Optional[str] = None,
        source_id: Optional[str] = None,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        full_text: Optional[str] = None,
        tags=None,
        embed: bool = True,
    ) -> int:
        """Add a paper, or update it in place if source_id already exists (upsert)."""
        if not title or not title.strip():
            raise ValueError("title is required")

        authors_s = _norm_list(authors)
        tags_s = _norm_list(tags)
        now = time.time()

        embedding_blob = None
        if embed:
            text_for_embedding = " ".join(
                filter(None, [title, abstract, full_text[:2000] if full_text else None])
            )
            try:
                vec = self._get_embedder().embed(text_for_embedding)
                embedding_blob = _pack(vec)
            except Exception:
                embedding_blob = None  # storage still succeeds without semantic search

        with self._lock:
            existing = None
            if source_id:
                existing = self.conn.execute(
                    "SELECT id FROM papers WHERE source_id = ?", (source_id,)
                ).fetchone()

            if existing:
                pid = existing["id"]
                self.conn.execute(
                    """UPDATE papers SET title=?, authors=?, abstract=?, source=?, url=?,
                       file_path=?, full_text=?, tags=?, embedding=COALESCE(?, embedding),
                       updated_at=? WHERE id=?""",
                    (title, authors_s, abstract, source, url, file_path, full_text,
                     tags_s, embedding_blob, now, pid),
                )
            else:
                cur = self.conn.execute(
                    """INSERT INTO papers
                       (title, authors, abstract, source, source_id, url, file_path,
                        full_text, tags, embedding, added_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (title, authors_s, abstract, source, source_id, url, file_path,
                     full_text, tags_s, embedding_blob, now, now),
                )
                pid = cur.lastrowid
            self.conn.commit()
        return pid

    def get(self, id_: int, include_embedding: bool = False) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM papers WHERE id = ?", (id_,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row, include_embedding)

    def delete(self, id_: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM papers WHERE id = ?", (id_,))
            self.conn.commit()
        return cur.rowcount > 0

    def list(
        self, tag: Optional[str] = None, source: Optional[str] = None,
        limit: int = 50, offset: int = 0,
    ) -> list[dict]:
        clauses, params = [], []
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"SELECT * FROM papers {where} ORDER BY added_at DESC LIMIT ? OFFSET ?", params
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        by_source = {
            r["source"] or "unknown": r["n"]
            for r in self.conn.execute(
                "SELECT source, COUNT(*) AS n FROM papers GROUP BY source"
            ).fetchall()
        }
        embedded = self.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        return {"papers": total, "by_source": by_source, "embedded": embedded,
                "fts_available": self._fts_available}

    # -- search --------------------------------------------------------------

    def search(self, query: str, k: int = 10, semantic: bool = True) -> list[dict]:
        """Hybrid search: semantic (cosine over embeddings) blended with FTS5
        keyword matches when available; falls back to a lexical LIKE scan on
        SQLite builds without FTS5.
        """
        if not query or not query.strip():
            return []

        scores: dict[int, float] = {}

        if semantic:
            try:
                qvec = self._get_embedder().embed(query)
                rows = self.conn.execute(
                    "SELECT id, embedding FROM papers WHERE embedding IS NOT NULL"
                ).fetchall()
                for r in rows:
                    sim = _cosine(qvec, _unpack(r["embedding"]))
                    if sim > 0:
                        scores[r["id"]] = max(scores.get(r["id"], 0.0), sim)
            except Exception:
                pass  # semantic scoring is best-effort; keyword search still runs below

        if self._fts_available:
            try:
                fts_rows = self.conn.execute(
                    "SELECT rowid AS id, bm25(papers_fts) AS rank FROM papers_fts "
                    "WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?",
                    (query, max(k * 3, 30)),
                ).fetchall()
                if fts_rows:
                    worst = max(r["rank"] for r in fts_rows) or 1.0
                    for r in fts_rows:
                        # bm25() is lower-is-better and unbounded; fold it into a
                        # [0, 0.5] boost so it can nudge but not dominate cosine sim.
                        boost = 0.5 * (1.0 - (r["rank"] / worst if worst else 0.0))
                        scores[r["id"]] = scores.get(r["id"], 0.0) + boost
            except sqlite3.OperationalError:
                pass  # malformed FTS query syntax — fall through to LIKE below
        elif not scores:
            like = f"%{query}%"
            rows = self.conn.execute(
                "SELECT id FROM papers WHERE title LIKE ? OR abstract LIKE ? "
                "OR full_text LIKE ? LIMIT ?",
                (like, like, like, max(k * 3, 30)),
            ).fetchall()
            for r in rows:
                scores[r["id"]] = scores.get(r["id"], 0.0) + 0.1

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        results = []
        for pid, score in ranked:
            row = self.conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()
            if row:
                d = self._row_to_dict(row)
                d["score"] = round(score, 6)
                results.append(d)
        return results

    # -- internal --------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row, include_embedding: bool = False) -> dict:
        d = {k: row[k] for k in row.keys() if k != "embedding"}
        d["authors"] = [a.strip() for a in (row["authors"] or "").split(",") if a.strip()]
        d["tags"] = [t.strip() for t in (row["tags"] or "").split(",") if t.strip()]
        if include_embedding:
            d["embedding"] = _unpack(row["embedding"])
        return d

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
