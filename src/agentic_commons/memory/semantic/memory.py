"""Semantic memory — what is generally true (E14, LLD §8).

A structured fact store in SQLite, retrieved through the same cascade
discipline used everywhere else: exact key lookup, then a LIKE scan,
then fuzzy key matching, and only then an embedding model. For
well-structured facts the first tier answers almost everything, which is
why reaching for embeddings first is both slower and worse.

Supersession is recorded, never overwritten. The old fact stays readable
with a pointer to what replaced it — that history is what makes "why did
the agent believe that last week?" answerable.
"""

import difflib
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_facts (
    fact_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    category       TEXT NOT NULL,
    confidence     REAL NOT NULL,
    importance     REAL NOT NULL,
    source         TEXT NOT NULL,
    provenance     TEXT NOT NULL,
    confirmed      INTEGER NOT NULL,
    event_time     TEXT NOT NULL,
    ingestion_time TEXT NOT NULL,
    valid_until    TEXT,
    superseded_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_semantic_key ON semantic_facts (key);
"""

_INSERT = """
INSERT INTO semantic_facts (
    key, value, category, confidence, importance, source, provenance,
    confirmed, event_time, ingestion_time, valid_until, superseded_by
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class SemanticStore:
    """Structured facts with exact-then-fuzzy key lookup."""

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        settings = get_settings().memory.semantic
        self._pool = pool or SqliteConnectionPool()
        self._enabled = settings.enabled
        self._fuzzy_threshold = settings.fuzzy_match_threshold
        self._logger = get_logger("memory")
        self._pool.execute_script(_SCHEMA)

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Store a fact. Conflicts are the write pipeline's business, not ours."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="semantic memory disabled")
        stored = fact.model_copy(update={"memory_type": "semantic", "provenance": provenance})
        self._pool.execute(
            _INSERT,
            (
                stored.key,
                stored.value,
                stored.category,
                stored.confidence,
                stored.importance,
                provenance.source,
                provenance.kind,
                int(provenance.confirmed),
                stored.event_time.isoformat(),
                provenance.ingestion_time.isoformat(),
                stored.valid_until.isoformat() if stored.valid_until else None,
                stored.superseded_by,
            ),
        )
        return WriteResult(route="auto_commit", written=True, fact=stored, reason="committed")

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Facts for a query, cheapest retrieval tier first."""
        if query.key:
            exact = self.get(query.key, query.include_superseded)
            if exact:
                return exact[: query.limit]
            fuzzy = self.fuzzy_get(query.key, query.include_superseded)
            if fuzzy:
                return fuzzy[: query.limit]
        return self.search(query)[: query.limit]

    def get(self, key: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Tier 1 — exact key match, newest first."""
        sql = "SELECT * FROM semantic_facts WHERE key = ?"
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY fact_id DESC"
        return [_to_fact(row) for row in self._pool.iter_rows(sql, (key,))]

    def search(self, query: MemoryQuery) -> list[MemoryFact]:
        """Tier 2 — literal substring scan over keys and values."""
        clauses, params = ["1 = 1"], []
        if query.text:
            clauses.append("(value LIKE ? OR key LIKE ?)")
            params += [f"%{query.text}%", f"%{query.text}%"]
        if query.category:
            clauses.append("category = ?")
            params.append(query.category)
        if not query.include_superseded:
            clauses.append("superseded_by IS NULL")
        sql = f"SELECT * FROM semantic_facts WHERE {' AND '.join(clauses)} ORDER BY fact_id DESC"
        return [_to_fact(row) for row in self._pool.iter_rows(sql, tuple(params))]

    def fuzzy_get(self, key: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Tier 3 — closest stored key above the configured similarity floor.

        Still no model: sequence-ratio matching handles the realistic
        case (a near-miss key) at a fraction of the cost, and reports a
        similarity a human can check.
        """
        matches = difflib.get_close_matches(key, self.keys(), n=1, cutoff=self._fuzzy_threshold)
        if not matches:
            return []
        self._logger.debug("Semantic fuzzy key match", requested=key, matched=matches[0])
        return self.get(matches[0], include_superseded)

    def keys(self) -> list[str]:
        """Every distinct key currently held."""
        return [
            str(row["key"])
            for row in self._pool.iter_rows(
                "SELECT DISTINCT key FROM semantic_facts WHERE superseded_by IS NULL"
            )
        ]

    def supersede(self, key: str, superseded_by: str) -> int:
        """Mark existing facts for a key as replaced, keeping them readable."""
        result = self._pool.execute(
            "UPDATE semantic_facts SET superseded_by = ? WHERE key = ? AND superseded_by IS NULL",
            (superseded_by, key),
        )
        return result.affected

    def revoke(self, key: str) -> int:
        """Hard delete every fact for a key.

        Distinct from supersession on purpose (E16): a revocation request
        must actually remove the data, not annotate it. A "deletion" that
        leaves the value readable is a deletion failure.
        """
        result = self._pool.execute("DELETE FROM semantic_facts WHERE key = ?", (key,))
        self._logger.info("Revoked semantic facts", key=key, removed=result.affected)
        return result.affected

    def decay_unconfirmed(self, older_than_days: int | None = None) -> int:
        """Expire unconfirmed self-reported facts past their confirmation window.

        A security posture, not housekeeping: an unverified claim that
        never gets confirmed should lose its standing over time rather
        than harden into apparent fact through sheer persistence.
        """
        days = older_than_days or get_settings().memory.write_pipeline.decay_if_unconfirmed_days
        cutoff = utc_now().timestamp() - days * 86400
        expired = 0
        for fact in self.search(MemoryQuery(limit=10_000)):
            if fact.provenance.confirmed or fact.provenance.kind != "self_reported":
                continue
            if fact.provenance.ingestion_time.timestamp() < cutoff:
                expired += self.supersede(fact.key, "decayed_unconfirmed")
        return expired


def _to_fact(row: dict[str, Any]) -> MemoryFact:
    """Rehydrate a stored row into a MemoryFact."""
    return MemoryFact(
        key=row["key"],
        value=row["value"],
        memory_type="semantic",
        category=row["category"],
        confidence=row["confidence"],
        importance=row["importance"],
        provenance=Provenance(
            source=row["source"],
            kind=row["provenance"],
            confirmed=bool(row["confirmed"]),
            ingestion_time=row["ingestion_time"],
        ),
        event_time=row["event_time"],
        valid_until=row["valid_until"],
        superseded_by=row["superseded_by"],
    )
