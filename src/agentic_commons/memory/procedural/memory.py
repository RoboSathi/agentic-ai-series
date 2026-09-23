"""Procedural memory — versioned how-to artifacts (E14, LLD §8)."""

from typing import Any

from agentic_commons.logging import get_logger, logged
from agentic_commons.memory.types import MemoryFact, MemoryQuery, Provenance, WriteResult
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_procedures (
    agent_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    version       INTEGER NOT NULL,
    body          TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    source        TEXT NOT NULL,
    provenance    TEXT NOT NULL,
    rationale     TEXT NOT NULL,
    active        INTEGER NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (agent_id, name, version)
);
CREATE INDEX IF NOT EXISTS idx_platform_procedures_active
    ON platform_procedures (agent_id, name, active);
"""


class ProceduralStore:
    """Versioned procedures, with exactly one active version per agent/name."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._logger = get_logger("memory")
        self._store.execute_script(_SCHEMA)
        self._migrate_legacy_table()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Record a new version of a procedure and make it the active one."""
        version = self.latest_version(fact.key) + 1
        self._store.execute(
            "UPDATE platform_procedures SET active = 0 WHERE agent_id = ? AND name = ?",
            (self._agent_id, fact.key),
        )
        self._store.execute(
            "INSERT INTO platform_procedures "
            "(agent_id, name, version, body, artifact_type, source, provenance, rationale, active, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                self._agent_id,
                fact.key,
                version,
                fact.value,
                str(fact.metadata.get("artifact_type", "how_to")),
                provenance.source,
                provenance.kind,
                str(fact.metadata.get("rationale", "")),
                utc_now().isoformat(),
            ),
        )
        stored = fact.model_copy(
            update={
                "memory_type": "procedural",
                "provenance": provenance,
                "metadata": {**fact.metadata, "version": version},
            }
        )
        return WriteResult(
            route="auto_commit", written=True, fact=stored, reason=f"version {version}"
        )

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Active procedures matching the query."""
        if query.key:
            found = self.get(query.key)
            return [found] if found else []
        sql = "SELECT * FROM platform_procedures WHERE agent_id = ? AND active = 1"
        params: list[Any] = [self._agent_id]
        if query.text:
            sql += " AND (name LIKE ? OR body LIKE ?)"
            params += [f"%{query.text}%", f"%{query.text}%"]
        return [_to_fact(row) for row in self._store.iter_rows(sql, tuple(params))][: query.limit]

    def get(self, name: str, version: int | None = None) -> MemoryFact | None:
        """The active version of a procedure, or a specific historical one."""
        if version is None:
            row = self._store.query_one(
                "SELECT * FROM platform_procedures WHERE agent_id = ? AND name = ? AND active = 1",
                (self._agent_id, name),
            )
        else:
            row = self._store.query_one(
                "SELECT * FROM platform_procedures WHERE agent_id = ? AND name = ? AND version = ?",
                (self._agent_id, name, version),
            )
        return _to_fact(row) if row else None

    def history(self, name: str) -> list[MemoryFact]:
        """Every version of a procedure, newest first."""
        return [
            _to_fact(row)
            for row in self._store.iter_rows(
                "SELECT * FROM platform_procedures WHERE agent_id = ? AND name = ? "
                "ORDER BY version DESC",
                (self._agent_id, name),
            )
        ]

    def rollback(self, name: str, version: int) -> MemoryFact | None:
        """Reactivate an earlier version — the point of keeping them all."""
        target = self.get(name, version)
        if target is None:
            self._logger.warning("Rollback target not found", procedure=name, version=version)
            return None
        self._store.execute(
            "UPDATE platform_procedures SET active = 0 WHERE agent_id = ? AND name = ?",
            (self._agent_id, name),
        )
        self._store.execute(
            "UPDATE platform_procedures SET active = 1 "
            "WHERE agent_id = ? AND name = ? AND version = ?",
            (self._agent_id, name, version),
        )
        self._logger.info("Rolled back procedure", procedure=name, version=version)
        return target

    def latest_version(self, name: str) -> int:
        """Highest version number recorded for a procedure (0 if none)."""
        row = self._store.query_one(
            "SELECT MAX(version) AS v FROM platform_procedures WHERE agent_id = ? AND name = ?",
            (self._agent_id, name),
        )
        return int(row["v"]) if row and row["v"] is not None else 0

    def names(self) -> list[str]:
        """Every procedure with at least one version."""
        return [
            str(row["name"])
            for row in self._store.iter_rows(
                "SELECT DISTINCT name FROM platform_procedures WHERE agent_id = ? ORDER BY name",
                (self._agent_id,),
            )
        ]

    def _migrate_legacy_table(self) -> None:
        """Copy legacy single-agent procedures into the scoped table."""
        if not self._table_exists("procedures"):
            return
        columns = self._table_columns("procedures")
        if "agent_id" in columns:
            rows = self._store.iter_rows(
                "SELECT agent_id, name, version, body, artifact_type, source, provenance, rationale, active, recorded_at "
                "FROM procedures WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._store.iter_rows(
                "SELECT name, version, body, artifact_type, source, provenance, rationale, active, recorded_at "
                "FROM procedures"
            )
        for row in rows:
            values = (
                row.get("agent_id", self._agent_id),
                row["name"],
                row["version"],
                row["body"],
                row["artifact_type"],
                row["source"],
                row["provenance"],
                row["rationale"],
                row["active"],
                row["recorded_at"],
            )
            self._store.execute(
                "INSERT OR IGNORE INTO platform_procedures "
                "(agent_id, name, version, body, artifact_type, source, provenance, rationale, active, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def _table_exists(self, table: str) -> bool:
        row = self._store.query_one(
            "SELECT 1 AS hit FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        )
        return row is not None

    def _table_columns(self, table: str) -> set[str]:
        return {str(row["name"]) for row in self._store.iter_rows(f"PRAGMA table_info({table})")}


def _to_fact(row: dict[str, Any]) -> MemoryFact:
    """Rehydrate a stored procedure row."""
    return MemoryFact(
        key=row["name"],
        value=row["body"],
        memory_type="procedural",
        category=row["artifact_type"],
        provenance=Provenance(source=row["source"], kind=row["provenance"]),
        metadata={
            "version": row["version"],
            "artifact_type": row["artifact_type"],
            "rationale": row["rationale"],
            "active": bool(row["active"]),
        },
    )
