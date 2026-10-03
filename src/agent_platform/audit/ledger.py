"""Append-only audit ledgers (E33, LLD §5).

Live agents use DuckDB for shared cross-agent observability. A legacy SQLite
implementation remains for episode demos, but even that path uses the same
platform persistence contract rather than touching an agent's domain tables.
"""

import json
from typing import Any, Protocol, runtime_checkable

from agent_platform.audit.types import AuditEntry, Decision
from agent_platform.config import get_settings
from agent_platform.logging import get_logger, instrument, logged
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_audit_entries (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id          TEXT    NOT NULL,
    trace_id          TEXT    NOT NULL,
    actor             TEXT    NOT NULL,
    action            TEXT    NOT NULL,
    authority_context TEXT    NOT NULL,
    model_version     TEXT    NOT NULL,
    policy_version    TEXT    NOT NULL,
    data_version      TEXT    NOT NULL,
    decision          TEXT    NOT NULL,
    rationale         TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_platform_audit_trace
    ON platform_audit_entries (agent_id, trace_id, seq);
CREATE INDEX IF NOT EXISTS idx_platform_audit_decision
    ON platform_audit_entries (agent_id, decision, seq);
"""

_DUCKDB_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS platform_audit_seq START 1;
CREATE TABLE IF NOT EXISTS platform_audit_entries (
    seq               BIGINT PRIMARY KEY DEFAULT nextval('platform_audit_seq'),
    agent_id          VARCHAR NOT NULL,
    trace_id          VARCHAR NOT NULL,
    actor             VARCHAR NOT NULL,
    action            VARCHAR NOT NULL,
    authority_context VARCHAR NOT NULL,
    model_version     VARCHAR NOT NULL,
    policy_version    VARCHAR NOT NULL,
    data_version      VARCHAR NOT NULL,
    decision          VARCHAR NOT NULL,
    rationale         VARCHAR NOT NULL,
    timestamp         VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_platform_audit_trace
    ON platform_audit_entries (agent_id, trace_id, seq);
CREATE INDEX IF NOT EXISTS idx_platform_audit_decision
    ON platform_audit_entries (agent_id, decision, seq);
"""

_INSERT = """
INSERT INTO platform_audit_entries (
    agent_id, trace_id, actor, action, authority_context, model_version,
    policy_version, data_version, decision, rationale, timestamp
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@runtime_checkable
class AuditLedger(Protocol):
    """Append-only evidence chain over consequential decisions."""

    def record(self, entry: AuditEntry) -> None: ...

    def reconstruct(self, trace_id: str) -> list[AuditEntry]: ...


@instrument("audit")
class DuckDbAuditLedger:
    """Shared append-only audit ledger used by live agents."""

    def __init__(self, store: DuckDbObservabilityStore, agent_id: str | None = None) -> None:
        settings = get_settings().audit
        self._store = store
        self._agent_id = agent_id or get_settings().agent.id
        self._policy_version = settings.policy_version
        self._data_version = settings.data_version
        self._logger = get_logger("audit")
        self._store.execute_script(_DUCKDB_SCHEMA)

    @logged("audit")
    def record(self, entry: AuditEntry) -> None:
        """Append one shared audit event, stamping the owning agent."""
        stamped = entry.model_copy(
            update={
                "agent_id": entry.agent_id or self._agent_id,
                "policy_version": entry.policy_version or self._policy_version,
                "data_version": entry.data_version or self._data_version,
            }
        )
        self._store.execute(
            _INSERT,
            (
                stamped.agent_id,
                stamped.trace_id,
                stamped.actor,
                stamped.action,
                json.dumps(stamped.authority_context, default=str),
                stamped.model_version,
                stamped.policy_version,
                stamped.data_version,
                stamped.decision,
                stamped.rationale,
                stamped.timestamp.isoformat(),
            ),
        )

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        """Every event for the current agent and trace, in causal order."""
        return [
            self._to_entry(row)
            for row in self._store.iter_rows(
                "SELECT * FROM platform_audit_entries "
                "WHERE agent_id = ? AND trace_id = ? ORDER BY seq ASC",
                (self._agent_id, trace_id),
            )
        ]

    def by_decision(self, decision: Decision) -> list[AuditEntry]:
        """Every shared audit event for the current agent and verdict."""
        return [
            self._to_entry(row)
            for row in self._store.iter_rows(
                "SELECT * FROM platform_audit_entries "
                "WHERE agent_id = ? AND decision = ? ORDER BY seq ASC",
                (self._agent_id, decision),
            )
        ]

    def count(self) -> int:
        """Total audit events for the current agent."""
        row = self._store.query_one(
            "SELECT COUNT(*) AS n FROM platform_audit_entries WHERE agent_id = ?",
            (self._agent_id,),
        )
        return int(row["n"]) if row else 0

    @staticmethod
    def _to_entry(row: dict[str, Any]) -> AuditEntry:
        data = dict(row)
        data.pop("seq", None)
        data["authority_context"] = json.loads(data["authority_context"])
        return AuditEntry.model_validate(data)


@instrument("audit")
class SqliteAuditLedger:
    """Legacy SQLite audit ledger over platform-owned state."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        settings = get_settings().audit
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._policy_version = settings.policy_version
        self._data_version = settings.data_version
        self._logger = get_logger("audit")
        self._store.execute_script(_SQLITE_SCHEMA)
        self._migrate_legacy_table()

    @logged("audit")
    def record(self, entry: AuditEntry) -> None:
        """Append one entry. Versions and agent identity default from config."""
        stamped = entry.model_copy(
            update={
                "agent_id": entry.agent_id or self._agent_id,
                "policy_version": entry.policy_version or self._policy_version,
                "data_version": entry.data_version or self._data_version,
            }
        )
        self._store.execute(
            _INSERT,
            (
                stamped.agent_id,
                stamped.trace_id,
                stamped.actor,
                stamped.action,
                json.dumps(stamped.authority_context, default=str),
                stamped.model_version,
                stamped.policy_version,
                stamped.data_version,
                stamped.decision,
                stamped.rationale,
                stamped.timestamp.isoformat(),
            ),
        )

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        return [
            self._to_entry(row)
            for row in self._store.iter_rows(
                "SELECT * FROM platform_audit_entries WHERE agent_id = ? AND trace_id = ? "
                "ORDER BY seq ASC",
                (self._agent_id, trace_id),
            )
        ]

    def by_decision(self, decision: Decision) -> list[AuditEntry]:
        return [
            self._to_entry(row)
            for row in self._store.iter_rows(
                "SELECT * FROM platform_audit_entries WHERE agent_id = ? AND decision = ? "
                "ORDER BY seq ASC",
                (self._agent_id, decision),
            )
        ]

    def count(self) -> int:
        row = self._store.query_one(
            "SELECT COUNT(*) AS n FROM platform_audit_entries WHERE agent_id = ?",
            (self._agent_id,),
        )
        return int(row["n"]) if row else 0

    def _migrate_legacy_table(self) -> None:
        """Copy legacy SQLite audit rows into the platform-owned table."""
        if not self._table_exists("audit_entries"):
            return
        columns = self._table_columns("audit_entries")
        if "agent_id" in columns:
            rows = self._store.iter_rows(
                "SELECT agent_id, trace_id, actor, action, authority_context, model_version, "
                "policy_version, data_version, decision, rationale, timestamp "
                "FROM audit_entries WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._store.iter_rows(
                "SELECT trace_id, actor, action, authority_context, model_version, "
                "policy_version, data_version, decision, rationale, timestamp FROM audit_entries"
            )
        for row in rows:
            values = (
                row.get("agent_id", self._agent_id),
                row["trace_id"],
                row["actor"],
                row["action"],
                row["authority_context"],
                row["model_version"],
                row["policy_version"],
                row["data_version"],
                row["decision"],
                row["rationale"],
                row["timestamp"],
            )
            exists = self._store.query_one(
                "SELECT 1 AS hit FROM platform_audit_entries WHERE agent_id = ? "
                "AND trace_id = ? AND actor = ? AND action = ? AND timestamp = ? LIMIT 1",
                (self._agent_id, values[1], values[2], values[3], values[10]),
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

    @staticmethod
    def _to_entry(row: dict[str, Any]) -> AuditEntry:
        data = dict(row)
        data.pop("seq", None)
        data["authority_context"] = json.loads(data["authority_context"])
        return AuditEntry.model_validate(data)


class NullAuditLedger:
    """No-op ledger for episodes that deliberately disable audit."""

    def record(self, entry: AuditEntry) -> None:
        return None

    def reconstruct(self, trace_id: str) -> list[AuditEntry]:
        return []


def create_audit_ledger(
    pool: SqliteConnectionPool | PlatformStore | None = None,
    *,
    store: DuckDbObservabilityStore | None = None,
    agent_id: str | None = None,
) -> AuditLedger:
    """Config-driven audit factory."""
    if not get_settings().audit.enabled:
        return NullAuditLedger()
    if store is not None:
        return DuckDbAuditLedger(store, agent_id=agent_id)
    if pool is not None:
        return SqliteAuditLedger(pool, agent_id=agent_id)
    return DuckDbAuditLedger(DuckDbObservabilityStore(), agent_id=agent_id)
