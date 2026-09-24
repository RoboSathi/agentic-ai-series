"""Security quarantine queue for requests that must not reach the model.

The queue is deliberately part of the resilience package: a rejected request
must have a durable destination, otherwise a security detector merely creates
an observable warning while the workflow continues.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_commons.logging import get_logger, instrument
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

DeadLetterReason = Literal["prompt_injection", "content_moderation", "security_policy"]
DeadLetterStatus = Literal["pending", "in_review", "resolved", "discarded"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_resilience_dead_letter_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL,
    trace_id        TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    principal_id    TEXT,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    payload         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    reviewer        TEXT,
    resolution      TEXT
);
CREATE INDEX IF NOT EXISTS idx_resilience_dlq_pending
    ON platform_resilience_dead_letter_queue (agent_id, status, created_at);
"""


class DeadLetterEntry(BaseModel):
    """One quarantined request waiting for operational review."""

    id: int | None = None
    agent_id: str
    trace_id: str = ""
    session_id: str = ""
    # Whoever the quarantined request was on behalf of. A string, and
    # called `principal` rather than `customer`, because the platform
    # quarantines requests for agents that have no customers.
    principal_id: str | None = None
    reason: DeadLetterReason
    status: DeadLetterStatus = "pending"
    payload: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    reviewer: str | None = None
    resolution: str | None = None


@instrument("resilience")
class DeadLetterQueue:
    """Durable quarantine queue for security-rejected requests.

    Payloads are expected to be sanitized by the caller before enqueueing.
    The queue never executes or forwards them to an LLM.
    """

    def __init__(
        self,
        store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        agent_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._store = as_platform_store(store, agent_id=agent_id)
        self._agent_id = agent_id or self._store.agent_id
        self._enabled = enabled
        self._logger = get_logger("resilience")
        self._store.execute_script(_SCHEMA)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enqueue(
        self,
        *,
        reason: DeadLetterReason,
        payload: str,
        trace_id: str = "",
        session_id: str = "",
        principal_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeadLetterEntry | None:
        """Persist a rejected request for later human review."""
        if not self._enabled:
            return None
        now = datetime.now(timezone.utc).isoformat()
        self._store.execute(
            """
            INSERT INTO platform_resilience_dead_letter_queue (
                agent_id, trace_id, session_id, principal_id, reason, status,
                payload, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            """,
            (
                self._agent_id,
                trace_id,
                session_id,
                principal_id,
                reason,
                payload,
                _json(metadata or {}),
                now,
                now,
            ),
        )
        row = self._store.query_one(
            "SELECT * FROM platform_resilience_dead_letter_queue "
            "WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
            (self._agent_id,),
        )
        if row is None:
            self._logger.error("Dead-letter enqueue succeeded without a readable row")
            return None
        return _entry(row)

    def pending(self) -> list[DeadLetterEntry]:
        """Return quarantined requests awaiting a human review decision."""
        return [
            _entry(row)
            for row in self._store.iter_rows(
                "SELECT * FROM platform_resilience_dead_letter_queue "
                "WHERE agent_id = ? AND status = 'pending' ORDER BY id",
                (self._agent_id,),
            )
        ]

    def mark_in_review(self, entry_id: int, reviewer: str) -> bool:
        """Claim one queued request for a human reviewer."""
        result = self._store.execute(
            "UPDATE platform_resilience_dead_letter_queue "
            "SET status = 'in_review', reviewer = ?, updated_at = ? "
            "WHERE id = ? AND agent_id = ? AND status = 'pending'",
            (reviewer, datetime.now(timezone.utc).isoformat(), entry_id, self._agent_id),
        )
        return result.affected > 0

    def resolve(self, entry_id: int, reviewer: str, resolution: str, *, discarded: bool = False) -> bool:
        """Close a reviewed queue item without executing its original payload."""
        status = "discarded" if discarded else "resolved"
        result = self._store.execute(
            "UPDATE platform_resilience_dead_letter_queue "
            "SET status = ?, reviewer = ?, resolution = ?, updated_at = ? "
            "WHERE id = ? AND agent_id = ? AND status IN ('pending', 'in_review')",
            (
                status,
                reviewer,
                resolution,
                datetime.now(timezone.utc).isoformat(),
                entry_id,
                self._agent_id,
            ),
        )
        return result.affected > 0


def _json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, default=str, sort_keys=True)


def _entry(row: dict[str, Any]) -> DeadLetterEntry:
    import json

    return DeadLetterEntry(
        id=int(row["id"]),
        agent_id=str(row["agent_id"]),
        trace_id=str(row.get("trace_id") or ""),
        session_id=str(row.get("session_id") or ""),
        principal_id=(
            str(row["principal_id"]) if row.get("principal_id") is not None else None
        ),
        reason=row["reason"],
        status=row["status"],
        payload=str(row["payload"]),
        metadata=json.loads(row.get("metadata") or "{}"),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        reviewer=row.get("reviewer"),
        resolution=row.get("resolution"),
    )
