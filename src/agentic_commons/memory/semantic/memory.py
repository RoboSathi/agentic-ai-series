"""Semantic memory — what is generally true (E14, LLD §8).

Semantic memory is platform-owned agent state. The storage schema is scoped
by ``agent_id`` because the same common library may host different agents in
the same SQLite file, and by ``subject_id`` because one agent holds facts
about many customers and a session may read only its own subject's.

It is a source of truth, so nothing is ever overwritten:

  * an update is a new **version** that supersedes the old one;
  * a delete is a **tombstone** — out of recall, still in history;
  * only an **erase** removes a row, for data-protection requests.

Reads are tiered cheapest-first (R1 exact → R2 FTS5 BM25), and recall
**reinforces** what it used (`access_count`, `last_accessed_at`).
"""

import difflib
import re
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

# Columns added after the table first shipped: migrated in place, with the
# defaults an existing row should have had all along.
_ADDED_COLUMNS = {
    "subject_id": "TEXT NOT NULL DEFAULT ''",
    "version": "INTEGER NOT NULL DEFAULT 1",
    "access_count": "INTEGER NOT NULL DEFAULT 0",
    "last_accessed_at": "TEXT",
    "deleted_at": "TEXT",
    "deleted_by": "TEXT",
}

# Full-text index (R2) kept in step with the facts table by triggers, so no
# write path can forget to update it.
_FTS_TABLE = "platform_semantic_fts"
_FTS_SCHEMA = f"""
CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5(
    key, value, content='platform_semantic_facts', content_rowid='fact_id'
);
CREATE TRIGGER IF NOT EXISTS platform_semantic_fts_ai AFTER INSERT ON platform_semantic_facts
BEGIN INSERT INTO {_FTS_TABLE}(rowid, key, value) VALUES (new.fact_id, new.key, new.value); END;
CREATE TRIGGER IF NOT EXISTS platform_semantic_fts_ad AFTER DELETE ON platform_semantic_facts
BEGIN INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, key, value)
    VALUES ('delete', old.fact_id, old.key, old.value); END;
INSERT INTO {_FTS_TABLE}({_FTS_TABLE}) VALUES ('rebuild');
"""

_INSERT = """
INSERT INTO platform_semantic_facts (
    agent_id, key, value, category, confidence, importance, source, provenance,
    confirmed, event_time, ingestion_time, valid_until, superseded_by,
    subject_id, version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Believed now: neither replaced by a newer version nor tombstoned.
_CURRENT = "superseded_by IS NULL AND deleted_at IS NULL"
# FTS5 query terms: plain word characters only, so nothing a customer types
# is ever interpreted as FTS syntax.
_FTS_TERM = re.compile(r"[A-Za-z0-9]{2,}")


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
        self._migrate_columns()
        self._ensure_fts()
        self._migrate_legacy_table()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def purge(self) -> int:
        """Remove every one of this agent's rows. Operator reset only.

        Deliberately outside the write pipeline: this is `make clean-memory`
        wiping a demo back to empty, not a memory operation with a
        principal — the same standing as rebuilding the database.
        """
        result = self._store.execute(
            "DELETE FROM platform_semantic_facts WHERE agent_id = ?", (self._agent_id,)
        )
        # The delete trigger keeps the index in step row by row; a rebuild
        # makes sure nothing is left behind whatever state it was in.
        self._store.execute(f"INSERT INTO {_FTS_TABLE}({_FTS_TABLE}) VALUES ('rebuild')")
        return result.affected

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
                stored.subject_id,
                stored.version,
            ),
        )
        return WriteResult(route="auto_commit", written=True, fact=stored, reason="committed")

    def latest(self, key: str, subject_id: str = "") -> MemoryFact | None:
        """The current version of one subject's fact, if there is one."""
        row = self._store.query_one(
            f"SELECT * FROM platform_semantic_facts WHERE agent_id = ? AND subject_id = ? "
            f"AND key = ? AND {_CURRENT} ORDER BY fact_id DESC LIMIT 1",
            (self._agent_id, subject_id, key),
        )
        return _to_fact(row) if row else None

    def exact(
        self, keys: list[str], identifiers: list[str], subject_id: str, limit: int
    ) -> list[MemoryFact]:
        """R1 — facts whose key is asked for, or whose value names an identifier."""
        if not keys and not identifiers:
            return []
        matches = ["key = ?" for _ in keys] + ["instr(value, ?) > 0" for _ in identifiers]
        sql = (
            "SELECT * FROM platform_semantic_facts WHERE agent_id = ? AND subject_id = ? "
            f"AND {_CURRENT} AND ({' OR '.join(matches)}) ORDER BY fact_id DESC LIMIT ?"
        )
        params = (self._agent_id, subject_id, *keys, *identifiers, limit)
        return [_to_fact(row) for row in self._store.iter_rows(sql, params)]

    def lexical(self, text: str, subject_id: str, limit: int) -> list[tuple[MemoryFact, float]]:
        """R2 — FTS5 BM25 over keys and values, best first, with a [0, 1] score."""
        terms = _FTS_TERM.findall(text)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = list(
            self._store.iter_rows(
                f"SELECT f.*, bm25({_FTS_TABLE}) AS rank FROM {_FTS_TABLE} "
                f"JOIN platform_semantic_facts f ON f.fact_id = {_FTS_TABLE}.rowid "
                f"WHERE {_FTS_TABLE} MATCH ? AND f.agent_id = ? AND f.subject_id = ? "
                f"AND f.superseded_by IS NULL AND f.deleted_at IS NULL "
                "ORDER BY rank LIMIT ?",
                (expression, self._agent_id, subject_id, limit),
            )
        )
        # bm25() is negative, more negative is better; scale against the best.
        best = min((float(row["rank"]) for row in rows), default=0.0)
        return [(_to_fact(row), float(row["rank"]) / best if best else 0.0) for row in rows]

    def tombstone(self, key: str, subject_id: str, actor: str) -> int:
        """Delete a fact the way a source of truth deletes: keep it, stop believing it."""
        result = self._store.execute(
            f"UPDATE platform_semantic_facts SET deleted_at = ?, deleted_by = ? "
            f"WHERE agent_id = ? AND subject_id = ? AND key = ? AND {_CURRENT}",
            (utc_now().isoformat(), actor, self._agent_id, subject_id, key),
        )
        return result.affected

    def erase(self, key: str, subject_id: str) -> int:
        """Remove every version of a fact — data-protection erasure, not a delete."""
        result = self._store.execute(
            "DELETE FROM platform_semantic_facts WHERE agent_id = ? AND subject_id = ? AND key = ?",
            (self._agent_id, subject_id, key),
        )
        self._logger.info("Erased semantic fact", key=key, removed=result.affected)
        return result.affected

    def reinforce(self, facts: list[MemoryFact]) -> None:
        """Record that recall used these facts (frequency and recency of use)."""
        now = utc_now().isoformat()
        for fact in facts:
            self._store.execute(
                "UPDATE platform_semantic_facts SET access_count = access_count + 1, "
                f"last_accessed_at = ? WHERE agent_id = ? AND subject_id = ? AND key = ? AND {_CURRENT}",
                (now, self._agent_id, fact.subject_id, fact.key),
            )

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
            sql += f" AND {_CURRENT}"
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
        if query.subject_id:
            clauses.append("subject_id = ?")
            params.append(query.subject_id)
        if not query.include_superseded:
            clauses.append(_CURRENT)
        sql = (
            "SELECT * FROM platform_semantic_facts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY fact_id DESC"
        )
        return [_to_fact(row) for row in self._store.iter_rows(sql, tuple(params))]

    def fuzzy_get(self, key: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Tier 3 — closest stored key above the configured similarity floor."""
        matches = difflib.get_close_matches(key, self.keys(), n=1, cutoff=self._fuzzy_threshold)
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
                f"WHERE agent_id = ? AND {_CURRENT}",
                (self._agent_id,),
            )
        ]

    def supersede(self, key: str, superseded_by: str, subject_id: str | None = None) -> int:
        """Mark existing facts for a key as replaced, keeping them readable.

        With a subject, only that subject's fact; without one, every
        subject's (the legacy, agent-wide behaviour).
        """
        sql = (
            "UPDATE platform_semantic_facts SET superseded_by = ? "
            "WHERE agent_id = ? AND key = ? AND superseded_by IS NULL"
        )
        params: tuple[Any, ...] = (superseded_by, self._agent_id, key)
        if subject_id is not None:
            sql += " AND subject_id = ?"
            params = (*params, subject_id)
        return self._store.execute(sql, params).affected

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

    def _migrate_columns(self) -> None:
        """Add the columns later versions introduced to an older table."""
        present = self._table_columns("platform_semantic_facts")
        for column, definition in _ADDED_COLUMNS.items():
            if column not in present:
                self._store.execute(
                    f"ALTER TABLE platform_semantic_facts ADD COLUMN {column} {definition}"
                )

    def _ensure_fts(self) -> None:
        """Create the full-text index once, filled from whatever is already stored."""
        if not self._table_exists(_FTS_TABLE):
            self._store.execute_script(_FTS_SCHEMA)

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
                self._store.execute(_INSERT, (*values, "", 1))

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
        subject_id=row.get("subject_id") or "",
        version=int(row.get("version") or 1),
        access_count=int(row.get("access_count") or 0),
        last_accessed_at=row.get("last_accessed_at"),
        deleted_at=row.get("deleted_at"),
        deleted_by=row.get("deleted_by"),
    )
