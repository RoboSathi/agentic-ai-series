"""Semantic memory — what is generally true (E14, LLD §8).

Semantic memory is platform-owned agent state. The storage schema is scoped
by ``agent_id`` because the same common library may host different agents in
the same SQLite file.
"""

import difflib
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_semantic_facts (
    fact_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id       TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_platform_semantic_key
    ON platform_semantic_facts (agent_id, key);
"""

_INSERT = """
INSERT INTO platform_semantic_facts (
    agent_id, key, value, category, confidence, importance, source, provenance,
    confirmed, event_time, ingestion_time, valid_until, superseded_by
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@instrument("memory")
class SemanticStore:
    """Structured facts with exact-then-fuzzy key lookup, agent-scoped."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        settings = get_settings().memory.semantic
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._fuzzy_threshold = settings.fuzzy_match_threshold
        self._logger = get_logger("memory")
        self._store.execute_script(_SCHEMA)
        self._migrate_legacy_table()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Store a fact. Conflicts are the write pipeline's business, not ours."""
        stored = fact.model_copy(update={"memory_type": "semantic", "provenance": provenance})
        self._store.execute(
            _INSERT,
            (
                self._agent_id,
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
        sql = "SELECT * FROM platform_semantic_facts WHERE agent_id = ? AND key = ?"
        params: tuple[Any, ...] = (self._agent_id, key)
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY fact_id DESC"
        return [_to_fact(row) for row in self._store.iter_rows(sql, params)]

    def search(self, query: MemoryQuery) -> list[MemoryFact]:
        """Tier 2 — literal substring scan over keys and values."""
        clauses, params = ["agent_id = ?"], [self._agent_id]
        if query.text:
            clauses.append("(value LIKE ? OR key LIKE ?)")
            params += [f"%{query.text}%", f"%{query.text}%"]
        if query.category:
            clauses.append("category = ?")
            params.append(query.category)
        if not query.include_superseded:
            clauses.append("superseded_by IS NULL")
        sql = (
            "SELECT * FROM platform_semantic_facts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY fact_id DESC"
        )
        return [_to_fact(row) for row in self._store.iter_rows(sql, tuple(params))]

    def fuzzy_get(self, key: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Tier 3 — closest stored key above the configured similarity floor."""
        matches = difflib.get_close_matches(
            key, self.keys(), n=1, cutoff=self._fuzzy_threshold
        )
        if not matches:
            return []
        self._logger.debug("Semantic fuzzy key match", requested=key, matched=matches[0])
        return self.get(matches[0], include_superseded)

    def keys(self) -> list[str]:
        """Every distinct key currently held for this agent."""
        return [
            str(row["key"])
            for row in self._store.iter_rows(
                "SELECT DISTINCT key FROM platform_semantic_facts "
                "WHERE agent_id = ? AND superseded_by IS NULL",
                (self._agent_id,),
            )
        ]

    def supersede(self, key: str, superseded_by: str) -> int:
        """Mark existing facts for a key as replaced, keeping them readable."""
        result = self._store.execute(
            "UPDATE platform_semantic_facts SET superseded_by = ? "
            "WHERE agent_id = ? AND key = ? AND superseded_by IS NULL",
            (superseded_by, self._agent_id, key),
        )
        return result.affected

    def revoke(self, key: str) -> int:
        """Hard delete every fact for this agent and key."""
        result = self._store.execute(
            "DELETE FROM platform_semantic_facts WHERE agent_id = ? AND key = ?",
            (self._agent_id, key),
        )
        self._logger.info("Revoked semantic facts", key=key, removed=result.affected)
        return result.affected

    def decay_unconfirmed(self, older_than_days: int | None = None) -> int:
        """Expire unconfirmed self-reported facts past their confirmation window."""
        days = older_than_days or get_settings().memory.write_pipeline.decay_if_unconfirmed_days
        cutoff = utc_now().timestamp() - days * 86400
        expired = 0
        for fact in self.search(MemoryQuery(limit=10_000)):
            if fact.provenance.confirmed or fact.provenance.kind != "self_reported":
                continue
            if fact.provenance.ingestion_time.timestamp() < cutoff:
                expired += self.supersede(fact.key, "decayed_unconfirmed")
        return expired

    def _migrate_legacy_table(self) -> None:
        """Copy legacy single-agent semantic facts into the scoped table."""
        if not self._table_exists("semantic_facts"):
            return
        columns = self._table_columns("semantic_facts")
        if "agent_id" in columns:
            rows = self._store.iter_rows(
                "SELECT agent_id, key, value, category, confidence, importance, source, provenance, "
                "confirmed, event_time, ingestion_time, valid_until, superseded_by "
                "FROM semantic_facts WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._store.iter_rows(
                "SELECT key, value, category, confidence, importance, source, provenance, "
                "confirmed, event_time, ingestion_time, valid_until, superseded_by "
                "FROM semantic_facts"
            )
        for row in rows:
            values = (
                row.get("agent_id", self._agent_id),
                row["key"],
                row["value"],
                row["category"],
                row["confidence"],
                row["importance"],
                row["source"],
                row["provenance"],
                row["confirmed"],
                row["event_time"],
                row["ingestion_time"],
                row["valid_until"],
                row["superseded_by"],
            )
            exists = self._store.query_one(
                "SELECT 1 AS hit FROM platform_semantic_facts WHERE agent_id = ? "
                "AND key = ? AND value = ? AND event_time = ? LIMIT 1",
                (self._agent_id, values[1], values[2], values[9]),
            )
            if exists is None:
                self._store.execute(_INSERT, values)

    def _table_exists(self, table: str) -> bool:
        row = self._store.query_one(
            "SELECT 1 AS hit FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        )
        return row is not None

    def _table_columns(self, table: str) -> set[str]:
        return {str(row["name"]) for row in self._store.iter_rows(f"PRAGMA table_info({table})")}


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
