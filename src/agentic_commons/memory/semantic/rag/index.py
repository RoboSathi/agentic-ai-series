"""The knowledge index — documents, passages and vectors in one SQLite file.

sqlite-vec rather than a vector server: the platform already persists in
SQLite, an agent's index is then one file at a path it chooses, and the
passages BM25 reads sit next to the vectors KNN reads, so the two legs of
hybrid retrieval can never disagree about what the corpus contains.

The vector table's width is fixed when it is created, from the index
fingerprint. That is deliberate: a vector of the wrong width is refused
by the table itself rather than silently compared across spaces.
"""

import json
import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from agentic_commons.memory.semantic.rag.types import Document, IndexFingerprint, Passage
from agentic_commons.observability.types import utc_now

_FINGERPRINT_KEY = "fingerprint"
_LAST_SYNCED_KEY = "last_synced"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_documents (
    path         TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    ingested_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_passages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    passage_id     TEXT NOT NULL UNIQUE,
    document_path  TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    heading_path   TEXT NOT NULL,
    content        TEXT NOT NULL,
    token_estimate INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_passages_document ON rag_passages (document_path);
"""

_DROP = """
DROP TABLE IF EXISTS rag_vectors;
DROP TABLE IF EXISTS rag_passages;
DROP TABLE IF EXISTS rag_documents;
DELETE FROM rag_meta;
"""

_INSERT_PASSAGE = """
INSERT INTO rag_passages
    (passage_id, document_path, ordinal, heading_path, content, token_estimate)
VALUES (?, ?, ?, ?, ?, ?)
"""


class SqliteVecIndex:
    """`VectorIndex` over one SQLite file with the sqlite-vec extension."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    def fingerprint(self) -> IndexFingerprint | None:
        value = self._meta(_FINGERPRINT_KEY)
        return IndexFingerprint.model_validate_json(value) if value else None

    def reset(self, fingerprint: IndexFingerprint | None) -> None:
        """Empty the index; with a fingerprint, size the vector table for it."""
        with self._lock, self._conn:
            self._conn.executescript(_DROP + _SCHEMA)
            if fingerprint is not None:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE rag_vectors USING vec0("
                    f"embedding float[{int(fingerprint.dimensions)}] distance_metric=cosine)"
                )
                self._set_meta(_FINGERPRINT_KEY, fingerprint.model_dump_json())

    def document_hashes(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT path, content_hash FROM rag_documents").fetchall()
        return {row["path"]: row["content_hash"] for row in rows}

    def upsert_document(
        self, document: Document, passages: list[Passage], vectors: list[list[float]]
    ) -> None:
        """Replace one document's passages and vectors in a single transaction."""
        with self._lock, self._conn:
            self._delete(document.path)
            for passage, vector in zip(passages, vectors, strict=True):
                cursor = self._conn.execute(_INSERT_PASSAGE, _passage_row(passage))
                self._conn.execute(
                    "INSERT INTO rag_vectors (rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(vector)),
                )
            self._conn.execute(
                "INSERT INTO rag_documents (path, content_hash, ingested_at) VALUES (?, ?, ?)",
                (document.path, document.content_hash, utc_now().isoformat()),
            )

    def delete_document(self, path: str) -> None:
        with self._lock, self._conn:
            self._delete(path)

    def passages(self) -> list[Passage]:
        rows = self._conn.execute(
            "SELECT * FROM rag_passages ORDER BY document_path, ordinal"
        ).fetchall()
        return [_passage_from(row) for row in rows]

    def nearest(self, vector: list[float], k: int) -> list[str]:
        """Passage ids by ascending cosine distance to the vector."""
        if not self._has_vectors():
            return []
        rows = self._conn.execute(
            "SELECT p.passage_id FROM ("
            "  SELECT rowid, distance FROM rag_vectors WHERE embedding MATCH ? AND k = ?"
            ") v JOIN rag_passages p ON p.id = v.rowid ORDER BY v.distance",
            (sqlite_vec.serialize_float32(vector), k),
        ).fetchall()
        return [row["passage_id"] for row in rows]

    def mark_synced(self) -> None:
        with self._lock, self._conn:
            self._set_meta(_LAST_SYNCED_KEY, utc_now().isoformat())

    def last_synced(self) -> str | None:
        return self._meta(_LAST_SYNCED_KEY)

    def _delete(self, path: str) -> None:
        if self._has_vectors():
            self._conn.execute(
                "DELETE FROM rag_vectors WHERE rowid IN "
                "(SELECT id FROM rag_passages WHERE document_path = ?)",
                (path,),
            )
        self._conn.execute("DELETE FROM rag_passages WHERE document_path = ?", (path,))
        self._conn.execute("DELETE FROM rag_documents WHERE path = ?", (path,))

    def _has_vectors(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'rag_vectors'"
        ).fetchone()
        return row is not None

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM rag_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO rag_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def _passage_row(passage: Passage) -> tuple[str, str, int, str, str, int]:
    return (
        passage.passage_id,
        passage.document_path,
        passage.ordinal,
        json.dumps(list(passage.heading_path)),
        passage.content,
        passage.token_estimate,
    )


def _passage_from(row: sqlite3.Row) -> Passage:
    return Passage(
        passage_id=row["passage_id"],
        document_path=row["document_path"],
        ordinal=row["ordinal"],
        heading_path=tuple(json.loads(row["heading_path"])),
        content=row["content"],
        token_estimate=row["token_estimate"],
    )
