"""Idempotency guard (LLD §13).

The durable record is platform operational state. It lives in the shared
SQLite file, but only through ``PlatformStore`` and is always scoped by
``agent_id`` so one agent cannot replay another agent's request key.
"""

import json
from typing import Any, Awaitable, Callable, Protocol, TypeVar, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.persistence import PlatformStore, SqliteConnectionPool, as_platform_store
from agent_platform.primitives.clock import utc_now

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_idempotency_records (
    agent_id    TEXT NOT NULL,
    key         TEXT NOT NULL,
    result_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, key)
);
CREATE INDEX IF NOT EXISTS idx_platform_idempotency_agent_time
    ON platform_idempotency_records (agent_id, recorded_at);
"""


@runtime_checkable
class IdempotencyGuard(Protocol):
    """Durable idempotency contract shared by resilience and Executor."""

    async def once(self, key: str, fn: Callable[[], Awaitable[T]]) -> T: ...

    def recorded_result(self, key: str) -> Any | None: ...

    def seen(self, key: str) -> bool: ...

    def remember(self, key: str, result: Any) -> None: ...

    def forget(self, key: str) -> None: ...


class SqliteIdempotencyGuard:
    """Durable at-most-once execution scoped to one agent."""

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._enabled = get_settings().resilience.idempotency.enabled
        self._logger = get_logger("resilience")
        self._store.execute_script(_SCHEMA)
        self._migrate_legacy_table()

    @property
    def agent_id(self) -> str:
        """Agent whose idempotency namespace this guard enforces."""
        return self._agent_id

    async def once(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Run the operation, or replay the recorded result for this key."""
        if not self._enabled:
            return await fn()
        recorded = self.recorded_result(key)
        if recorded is not None:
            self._logger.info("Replaying recorded result for duplicate request", key=key)
            return recorded  # type: ignore[no-any-return]
        result = await fn()
        self._record(key, result)
        return result

    def recorded_result(self, key: str) -> Any | None:
        """The stored outcome for a key, if this request was already handled."""
        row = self._store.query_one(
            "SELECT result_json FROM platform_idempotency_records WHERE agent_id = ? AND key = ?",
            (self._agent_id, key),
        )
        return json.loads(row["result_json"]) if row else None

    def seen(self, key: str) -> bool:
        """Whether a key has already been handled."""
        return (
            self._store.query_one(
                "SELECT 1 AS hit FROM platform_idempotency_records WHERE agent_id = ? AND key = ?",
                (self._agent_id, key),
            )
            is not None
        )

    def remember(self, key: str, result: Any) -> None:
        """Record an outcome for a key the caller executed itself."""
        self._record(key, result)

    def forget(self, key: str) -> None:
        """Release a key whose operation was undone, so a new request runs rather than replays.

        A remembered result answers "this was already done". Once the effect
        has been reversed — a payment voided before it settled — that answer
        is false, and the next identical request is a new one.
        """
        self._store.execute(
            "DELETE FROM platform_idempotency_records WHERE agent_id = ? AND key = ?",
            (self._agent_id, key),
        )

    def _record(self, key: str, result: Any) -> None:
        """Persist the outcome so a replay can return it verbatim."""
        self._store.execute(
            "INSERT OR REPLACE INTO platform_idempotency_records "
            "(agent_id, key, result_json, recorded_at) VALUES (?, ?, ?, ?)",
            (self._agent_id, key, json.dumps(result, default=_serialize), utc_now().isoformat()),
        )

    def _migrate_legacy_table(self) -> None:
        """Copy legacy idempotency rows into the agent-scoped platform table."""
        if not self._table_exists("idempotency_records"):
            return
        columns = self._table_columns("idempotency_records")
        if "agent_id" in columns:
            rows = self._store.iter_rows(
                "SELECT agent_id, key, result_json, recorded_at FROM idempotency_records "
                "WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._store.iter_rows(
                "SELECT key, result_json, recorded_at FROM idempotency_records"
            )
        for row in rows:
            values = (
                row.get("agent_id", self._agent_id),
                row["key"],
                row["result_json"],
                row["recorded_at"],
            )
            self._store.execute(
                "INSERT OR IGNORE INTO platform_idempotency_records "
                "(agent_id, key, result_json, recorded_at) VALUES (?, ?, ?, ?)",
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


def _serialize(value: Any) -> Any:
    """Serialize pydantic models and anything else JSON cannot take directly."""
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else str(value)
