"""Idempotency guard (LLD §13).

"Exactly once" is not available over an unreliable network; "at least
once delivery plus idempotent handling" is. This guard is the second
half: a keyed record, written durably, so that the second arrival of the
same request returns the first outcome instead of repeating the effect.

The record is in SQLite rather than memory on purpose — a guard that
forgets on restart protects against retries and not against crashes,
which is the case that actually pays a customer twice.
"""

import json
from typing import Any, Awaitable, Callable, Protocol, TypeVar, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency_records (
    key         TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


@runtime_checkable
class IdempotencyGuard(Protocol):
    """Runs an operation at most once per key."""

    async def once(self, key: str, fn: Callable[[], Awaitable[T]]) -> T: ...


class SqliteIdempotencyGuard:
    """Durable at-most-once execution, keyed by the caller's request key."""

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        self._pool = pool or SqliteConnectionPool()
        self._enabled = get_settings().resilience.idempotency.enabled
        self._logger = get_logger("resilience")
        self._pool.execute_script(_SCHEMA)

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
        row = self._pool.query_one(
            "SELECT result_json FROM idempotency_records WHERE key = ?", (key,)
        )
        return json.loads(row["result_json"]) if row else None

    def seen(self, key: str) -> bool:
        """Whether a key has already been handled."""
        return (
            self._pool.query_one("SELECT 1 AS hit FROM idempotency_records WHERE key = ?", (key,))
            is not None
        )

    def remember(self, key: str, result: Any) -> None:
        """Record an outcome for a key the caller executed itself.

        For callers that must decide *whether* the result is worth
        replaying — a saga records only a success, so a failed attempt
        can legitimately be retried rather than replaying the failure.
        """
        self._record(key, result)

    def _record(self, key: str, result: Any) -> None:
        """Persist the outcome so a replay can return it verbatim."""
        self._pool.execute(
            "INSERT OR REPLACE INTO idempotency_records (key, result_json, recorded_at) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(result, default=_serialize), utc_now().isoformat()),
        )


def _serialize(value: Any) -> Any:
    """Serialize pydantic models and anything else JSON cannot take directly."""
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else str(value)
