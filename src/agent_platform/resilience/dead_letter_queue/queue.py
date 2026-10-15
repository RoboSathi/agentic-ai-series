"""Security quarantine for requests that must not reach the model.

The queue is deliberately part of the resilience package: a rejected request
must have a durable destination, otherwise a security detector merely creates
an observable warning while the workflow continues. Its destination is the
agent's one review queue (ADR 0012): a quarantine is a `quarantine` item a
person decides with `/review` like any other, and is never re-run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.core.persistence import PlatformStore, SqliteConnectionPool
from agent_platform.core.primitives.clock import utc_now
from agent_platform.logging import get_logger, instrument
from agent_platform.review.queue import ReviewQueue
from agent_platform.review.types import ReviewItem, ReviewStatus

DeadLetterReason = Literal["prompt_injection", "content_moderation", "security_policy"]
DeadLetterStatus = ReviewStatus


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
    """Durable quarantine for security-rejected requests, as review items.

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
        self._queue = ReviewQueue(store, agent_id=agent_id)
        self._agent_id = self._queue.agent_id
        self._kind = get_settings().review.kinds.quarantine
        self._enabled = enabled
        self._logger = get_logger("resilience")

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
        item = self._queue.add(
            ReviewItem(
                kind=self._kind,
                session_id=session_id,
                subject_id=principal_id or "",
                summary=f"{reason}: {payload}",
                detail={
                    "reason": reason,
                    "payload": payload,
                    "trace_id": trace_id,
                    "principal_id": principal_id,
                    "metadata": metadata or {},
                },
            )
        )
        return _entry(item)

    def pending(self) -> list[DeadLetterEntry]:
        """Quarantined requests awaiting a human decision."""
        return [_entry(item) for item in self._queue.items(kinds=(self._kind,))]

    def entries(self, status: DeadLetterStatus | None, limit: int) -> list[DeadLetterEntry]:
        """The newest entries first, in one status or in all of them (None)."""
        items = self._queue.items(status=status, kinds=(self._kind,), limit=limit)
        return [_entry(item) for item in reversed(items)]

    def get(self, entry_id: int) -> DeadLetterEntry | None:
        item = self._queue.get(entry_id)
        return _entry(item) if item is not None and item.kind == self._kind else None

    def counts(self) -> dict[str, int]:
        """How many entries are in each status."""
        return self._queue.counts((self._kind,))


def _entry(item: ReviewItem) -> DeadLetterEntry:
    detail = item.detail
    principal = detail.get("principal_id")
    return DeadLetterEntry(
        id=item.id,
        agent_id=item.agent_id,
        trace_id=str(detail.get("trace_id", "")),
        session_id=item.session_id,
        principal_id=str(principal) if principal is not None else None,
        reason=detail.get("reason", "security_policy"),
        status=item.status,
        payload=str(detail.get("payload", "")),
        metadata=dict(detail.get("metadata", {})),
        created_at=item.created_at,
        updated_at=item.decided_at or item.created_at,
        reviewer=item.decided_by,
        resolution=item.comment,
    )
