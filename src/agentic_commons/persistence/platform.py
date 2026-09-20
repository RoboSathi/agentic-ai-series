"""Platform-owned operational persistence contracts.

The platform and each agent intentionally share the same physical SQLite
file in this phase, but ownership is logical: ``agentic_commons`` can only
persist through ``PlatformStore`` and therefore only touches platform-owned
tables. Agent/domain repositories keep using their own repository layer.

This is an architectural boundary, not another database. The shared SQLite
file makes cross-domain SQL joins possible when an agent explicitly needs
them, while the store interface prevents common code from importing an
agent's schema or repository classes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.persistence.database import SqliteConnectionPool
from agentic_commons.persistence.types import ExecutionResult

_PLATFORM_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_agents (
    agent_id    TEXT PRIMARY KEY,
    agent_name  TEXT NOT NULL,
    version     TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


@runtime_checkable
class PlatformStore(Protocol):
    """Persistence port for platform-owned operational state."""

    @property
    def agent_id(self) -> str: ...

    @property
    def db_path(self) -> str: ...

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> ExecutionResult: ...

    def execute_many(
        self, query: str, seq_of_params: list[tuple[Any, ...]]
    ) -> ExecutionResult: ...

    def execute_script(self, script: str) -> None: ...

    def query_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None: ...

    def iter_rows(self, query: str, params: tuple[Any, ...] = ()) -> Iterator[dict[str, Any]]: ...


class SqlitePlatformStore:
    """PlatformStore implementation backed by the shared SQLite file."""

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
        agent_version: str | None = None,
    ) -> None:
        settings = get_settings()
        self._pool = pool or SqliteConnectionPool()
        self._agent_id = agent_id or settings.agent.id
        self._agent_version = agent_version or settings.agent.version
        self._pool.execute_script(_PLATFORM_SCHEMA)
        self._register_agent()

    @property
    def agent_id(self) -> str:
        """Logical owner of platform state written through this store."""
        return self._agent_id

    @property
    def db_path(self) -> str:
        """Physical SQLite file used by the underlying pool."""
        return self._pool.db_path

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        return self._pool.execute(query, params)

    def execute_many(
        self, query: str, seq_of_params: list[tuple[Any, ...]]
    ) -> ExecutionResult:
        return self._pool.execute_many(query, seq_of_params)

    def execute_script(self, script: str) -> None:
        self._pool.execute_script(script)

    def query_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        return self._pool.query_one(query, params)

    def iter_rows(self, query: str, params: tuple[Any, ...] = ()) -> Iterator[dict[str, Any]]:
        return self._pool.iter_rows(query, params)

    def _register_agent(self) -> None:
        now = _utc_now_iso()
        self._pool.execute(
            """
            INSERT INTO platform_agents (
                agent_id, agent_name, version, enabled, created_at, updated_at
            ) VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                agent_name = excluded.agent_name,
                version = excluded.version,
                enabled = 1,
                updated_at = excluded.updated_at
            """,
            (self._agent_id, self._agent_id, self._agent_version, now, now),
        )


def as_platform_store(
    store: PlatformStore | SqliteConnectionPool | None = None,
    *,
    agent_id: str | None = None,
) -> PlatformStore:
    """Normalize legacy raw pools and new platform stores to the platform port."""
    if store is None:
        return SqlitePlatformStore(agent_id=agent_id)
    if isinstance(store, SqlitePlatformStore):
        return store
    if isinstance(store, SqliteConnectionPool):
        return SqlitePlatformStore(store, agent_id=agent_id)
    return store


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
