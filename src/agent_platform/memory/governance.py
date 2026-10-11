"""Memory governance: who may do what, and how much they are believed.

Semantic memory is a source of truth, so an operation on it is a
privileged act, not a storage call:

  * **Provenance is derived from the principal.** A customer's words are
    `self_reported` because a customer said them — the caller cannot pass
    a stronger provenance in. That closes the hole where any code path
    could label a claim "verified" by asking for it.
  * **Every operation is authorized** as `memory:<class>:<op>:<category>`
    against the principal's grants, through the `MemoryAuthorizer` port so
    `memory` never imports `security`.
  * **Writes that need a person are queued durably** — in the agent's one
    review queue — not in a list that dies with the process — a refused or doubtful claim is evidence, and
    evidence that vanishes on restart was never kept.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.memory.types import (
    MemoryOperation,
    MemoryPrincipal,
    MemoryType,
    Provenance,
    ProvenanceKind,
    WriteResult,
)
from agent_platform.observability.types import utc_now
from agent_platform.persistence import PlatformStore
from agent_platform.review.queue import ReviewQueue
from agent_platform.review.types import ReviewItem

# The category segment used when a command names none.
ANY_CATEGORY = "*"
# How much of a held write a queue list shows on its line.
_SUMMARY_CHARS = 160


class MemoryDecision(BaseModel):
    """Whether a principal may perform one memory operation, and why."""

    allowed: bool
    permission: str
    reason: str
    granted_by: str = ""


@runtime_checkable
class MemoryAuthorizer(Protocol):
    """Decides memory operations for a principal (implemented in `security`)."""

    def authorize(self, principal: MemoryPrincipal, permission: str) -> MemoryDecision: ...


def memory_permission(memory_type: MemoryType, operation: MemoryOperation, category: str) -> str:
    """The permission one operation requires: `memory:<class>:<op>:<category>`."""
    return f"memory:{memory_type}:{operation}:{category or ANY_CATEGORY}"


def provenance_for(principal: MemoryPrincipal, source: str = "") -> Provenance:
    """Provenance as the principal's role determines it — never as a caller asks."""
    kinds = get_settings().memory.governance.provenance_by_role
    kind: ProvenanceKind = kinds[principal.role]
    return Provenance(source=source or principal.actor, kind=kind, ingestion_time=utc_now())


class CuratorQueue:
    """Writes held for a human, as items in the agent's one review queue.

    A fact and a skill are kinds of review item like any other (ADR 0011):
    a curator decides them with `/queue` or the Review tab, with a
    comment, beside the agent's own escalations.
    """

    def __init__(self, store: PlatformStore) -> None:
        self._queue = ReviewQueue(store)
        kinds = get_settings().review.kinds
        self._kinds = (kinds.memory_fact, kinds.skill)

    def put(self, result: WriteResult, kind: str) -> int:
        """Hold one write for a curator; returns its item id."""
        fact = result.fact
        item = self._queue.add(
            ReviewItem(
                kind=kind,
                subject_id=fact.subject_id if fact else "",
                session_id=str(fact.metadata.get("session_id", "")) if fact else "",
                summary=f"{fact.key} = {fact.value}"[:_SUMMARY_CHARS] if fact else result.reason,
                detail={"write": result.model_dump(mode="json")},
            )
        )
        return int(item.id or 0)

    def pending(self) -> list[tuple[int, WriteResult]]:
        """Everything awaiting a curator, oldest first."""
        return [
            (int(item.id or 0), WriteResult.model_validate(item.detail["write"]))
            for item in self._queue.items(kinds=self._kinds)
        ]

    def purge(self) -> int:
        """Remove every one of this agent's memory items. Operator reset only."""
        return self._queue.purge(self._kinds)

    def decide(self, queue_id: int, status: str, curator: str) -> None:
        self._queue.decide(queue_id, "approved" if status == "approved" else "denied")


class InMemoryCuratorQueue:
    """The same queue for a pipeline built without a platform store (tests)."""

    def __init__(self) -> None:
        self._items: dict[int, tuple[str, WriteResult]] = {}

    def put(self, result: WriteResult, kind: str = "") -> int:
        queue_id = len(self._items) + 1
        self._items[queue_id] = ("pending", result)
        return queue_id

    def pending(self) -> list[tuple[int, WriteResult]]:
        return [(qid, item) for qid, (status, item) in self._items.items() if status == "pending"]

    def decide(self, queue_id: int, status: str, curator: str) -> None:
        if queue_id in self._items:
            self._items[queue_id] = (status, self._items[queue_id][1])
