"""Platform-owned operational persistence contracts.

The platform and each agent intentionally share the same physical SQLite
file in this phase. The agent-specific runtime authority lives on
``platform_agents`` and ``platform_agent_permissions`` so the policy remains
simple and can be loaded as one immutable snapshot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.persistence.database import SqliteConnectionPool
from agentic_commons.persistence.types import ExecutionResult

# An agent's limits are per *action class*, in a table of their own.
#
# They used to be three columns on `platform_agents` — max_refund_amount,
# refund_budget_limit, refund_budget_window_hours — which said this
# platform only runs agents that issue refunds, and only ever one kind of
# budgeted action each. A second write tool would have needed a schema
# change to the shared platform; a second agent would have inherited
# columns meaningless to it.
_PLATFORM_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_agents (
    agent_id                     TEXT PRIMARY KEY,
    agent_name                   TEXT NOT NULL,
    version                      TEXT NOT NULL,
    enabled                      INTEGER NOT NULL DEFAULT 1,
    role                         TEXT NOT NULL DEFAULT 'unconfigured',
    authority_rank               INTEGER NOT NULL DEFAULT 999,
    created_at                   TEXT NOT NULL,
    updated_at                   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS platform_agent_permissions (
    agent_id    TEXT NOT NULL,
    permission  TEXT NOT NULL,
    PRIMARY KEY (agent_id, permission),
    FOREIGN KEY (agent_id) REFERENCES platform_agents(agent_id)
);

CREATE TABLE IF NOT EXISTS platform_agent_budgets (
    agent_id            TEXT NOT NULL,
    action_class        TEXT NOT NULL,
    per_action_limit_cents INTEGER NOT NULL DEFAULT 0,
    window_limit_cents     INTEGER NOT NULL DEFAULT 0,
    window_hours        INTEGER NOT NULL DEFAULT 24,
    PRIMARY KEY (agent_id, action_class),
    FOREIGN KEY (agent_id) REFERENCES platform_agents(agent_id)
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
        self._ensure_agent_columns()
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

    def _ensure_agent_columns(self) -> None:
        """Upgrade the pre-Phase-1 platform_agents table in place."""
        row = self._pool.query_one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'platform_agents'"
        )
        if row is None:
            return
        columns = {
            str(item["name"])
            for item in self._pool.iter_rows("PRAGMA table_info(platform_agents)")
        }
        additions = [
            ("role", "TEXT NOT NULL DEFAULT 'unconfigured'"),
            ("authority_rank", "INTEGER NOT NULL DEFAULT 999"),
        ]
        for name, ddl in additions:
            if name not in columns:
                self._pool.execute(f"ALTER TABLE platform_agents ADD COLUMN {name} {ddl}")

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
