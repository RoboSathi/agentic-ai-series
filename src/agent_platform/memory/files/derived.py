"""The full-text index over a memory directory — derived, never the truth.

Grep answers "which file mentions this word?"; ranking a dozen facts
against a sentence wants BM25. So each store keeps a SQLite FTS5 index
beside its files, built *from* them: delete it and the next read rebuilds
it, edit a record by hand and the fingerprint changes and it is rebuilt.
Nothing is ever recorded only here.
"""

import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from agent_platform.persistence import SqliteConnectionPool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id  TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    key     TEXT NOT NULL,
    text    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    key, text, content='docs', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs
BEGIN INSERT INTO docs_fts(rowid, key, text) VALUES (new.rowid, new.key, new.text); END;
CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs
BEGIN INSERT INTO docs_fts(docs_fts, rowid, key, text)
    VALUES ('delete', old.rowid, old.key, old.text); END;
"""

_FINGERPRINT = "fingerprint"
# FTS5 query terms: plain word characters only, so nothing a person types is
# ever read as FTS syntax.
_FTS_TERM = re.compile(r"[A-Za-z0-9]{2,}")


class IndexedDoc(BaseModel):
    """One searchable memory: its id in the store, whose it is, and its words."""

    doc_id: str
    subject: str
    key: str
    text: str


class DerivedIndex:
    """BM25 search over one store's current memories."""

    def __init__(self, path: Path) -> None:
        self._pool = SqliteConnectionPool(path)
        self._pool.execute_script(_SCHEMA)

    def replace(self, doc: IndexedDoc) -> None:
        """Index one memory, replacing whatever was indexed under its id."""
        self.remove(doc.doc_id)
        self._pool.execute(
            "INSERT INTO docs (doc_id, subject, key, text) VALUES (?, ?, ?, ?)",
            (doc.doc_id, doc.subject, doc.key, doc.text),
        )

    def remove(self, doc_id: str) -> None:
        """Drop one memory from the index (a no-op if it was not there)."""
        self._pool.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))

    def search(self, text: str, subject: str, limit: int) -> list[tuple[str, float]]:
        """(key, score in [0, 1]) for one subject's memories, best first."""
        terms = _FTS_TERM.findall(text)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = list(
            self._pool.iter_rows(
                "SELECT d.key AS key, bm25(docs_fts) AS rank FROM docs_fts "
                "JOIN docs d ON d.rowid = docs_fts.rowid "
                "WHERE docs_fts MATCH ? AND d.subject = ? ORDER BY rank LIMIT ?",
                (expression, subject, limit),
            )
        )
        # bm25() is negative and more negative is better: scale against the best.
        best = min((float(row["rank"]) for row in rows), default=0.0)
        return [(str(row["key"]), float(row["rank"]) / best if best else 0.0) for row in rows]

    def fingerprint(self) -> str:
        """What the index was last built from ("" if it never was)."""
        row = self._pool.query_one("SELECT value FROM meta WHERE name = ?", (_FINGERPRINT,))
        return str(row["value"]) if row else ""

    def rebuild(self, docs: Iterable[IndexedDoc], fingerprint: str) -> int:
        """Throw the index away and build it again; returns how many memories it holds."""
        self._pool.execute("DELETE FROM docs")
        self._pool.execute("INSERT INTO docs_fts(docs_fts) VALUES ('rebuild')")
        count = 0
        for doc in docs:
            self.replace(doc)
            count += 1
        self.mark(fingerprint)
        return count

    def mark(self, fingerprint: str) -> None:
        """Record that the index matches the records as they now are."""
        self._pool.execute(
            "INSERT OR REPLACE INTO meta (name, value) VALUES (?, ?)", (_FINGERPRINT, fingerprint)
        )

    def close(self) -> None:
        self._pool.close()
