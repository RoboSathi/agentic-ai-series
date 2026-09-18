"""Procedural memory — how to do things (E14, LLD §8).

Versioned how-to artifacts: corrected mistakes, tool-use recipes, and —
once `prompt_compilation` lands — DSPy-compiled prompts, which are simply
another legitimate artifact type here (E21).

Versioning is the whole design. A procedure that is edited in place
cannot be rolled back when the "improvement" turns out to be a
regression, and cannot answer which version produced last month's
behaviour.
"""

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
CREATE TABLE IF NOT EXISTS procedures (
    name           TEXT NOT NULL,
    version        INTEGER NOT NULL,
    body           TEXT NOT NULL,
    artifact_type  TEXT NOT NULL,
    source         TEXT NOT NULL,
    provenance     TEXT NOT NULL,
    rationale      TEXT NOT NULL,
    active         INTEGER NOT NULL,
    recorded_at    TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
"""


class ProceduralStore:
    """Versioned procedures, with exactly one active version per name."""

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        self._pool = pool or SqliteConnectionPool()
        self._enabled = get_settings().memory.procedural.enabled
        self._logger = get_logger("memory")
        self._pool.execute_script(_SCHEMA)

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Record a new version of a procedure and make it the active one."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="procedural memory disabled")
        version = self.latest_version(fact.key) + 1
        self._pool.execute("UPDATE procedures SET active = 0 WHERE name = ?", (fact.key,))
        self._pool.execute(
            "INSERT INTO procedures (name, version, body, artifact_type, source, "
            "provenance, rationale, active, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
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
        sql = "SELECT * FROM procedures WHERE active = 1"
        params: tuple[Any, ...] = ()
        if query.text:
            sql += " AND (name LIKE ? OR body LIKE ?)"
            params = (f"%{query.text}%", f"%{query.text}%")
        return [_to_fact(row) for row in self._pool.iter_rows(sql, params)][: query.limit]

    def get(self, name: str, version: int | None = None) -> MemoryFact | None:
        """The active version of a procedure, or a specific historical one."""
        if version is None:
            row = self._pool.query_one(
                "SELECT * FROM procedures WHERE name = ? AND active = 1", (name,)
            )
        else:
            row = self._pool.query_one(
                "SELECT * FROM procedures WHERE name = ? AND version = ?", (name, version)
            )
        return _to_fact(row) if row else None

    def history(self, name: str) -> list[MemoryFact]:
        """Every version of a procedure, newest first."""
        return [
            _to_fact(row)
            for row in self._pool.iter_rows(
                "SELECT * FROM procedures WHERE name = ? ORDER BY version DESC", (name,)
            )
        ]

    def rollback(self, name: str, version: int) -> MemoryFact | None:
        """Reactivate an earlier version — the point of keeping them all."""
        target = self.get(name, version)
        if target is None:
            self._logger.warning("Rollback target not found", procedure=name, version=version)
            return None
        self._pool.execute("UPDATE procedures SET active = 0 WHERE name = ?", (name,))
        self._pool.execute(
            "UPDATE procedures SET active = 1 WHERE name = ? AND version = ?", (name, version)
        )
        self._logger.info("Rolled back procedure", procedure=name, version=version)
        return target

    def latest_version(self, name: str) -> int:
        """Highest version number recorded for a procedure (0 if none)."""
        row = self._pool.query_one(
            "SELECT MAX(version) AS v FROM procedures WHERE name = ?", (name,)
        )
        return int(row["v"]) if row and row["v"] is not None else 0

    def names(self) -> list[str]:
        """Every procedure with at least one version."""
        return [
            str(row["name"])
            for row in self._pool.iter_rows("SELECT DISTINCT name FROM procedures ORDER BY name")
        ]


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
