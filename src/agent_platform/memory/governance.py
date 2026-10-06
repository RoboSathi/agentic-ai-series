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
  * **Writes that need a person are queued durably**, not in a list that
    dies with the process — a refused or doubtful claim is evidence, and
    evidence that vanishes on restart was never kept.
"""

import json
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

# The category segment used when a command names none.
ANY_CATEGORY = "*"


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
    """Writes held for a human, persisted so a restart does not drop them."""

    _TABLE = "platform_memory_curator_queue"

    def __init__(self, store: PlatformStore) -> None:
        self._store = store
        self._agent_id = store.agent_id
        self._store.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
                queue_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id   TEXT NOT NULL,
                status     TEXT NOT NULL,
                payload    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decided_by TEXT,
                decided_at TEXT
            )
            """
        )

    def put(self, result: WriteResult) -> int:
        """Hold one write for a curator; returns its queue id."""
        self._store.execute(
            f"INSERT INTO {self._TABLE} (agent_id, status, payload, created_at) "
            "VALUES (?, 'pending', ?, ?)",
            (self._agent_id, result.model_dump_json(), utc_now().isoformat()),
        )
        row = self._store.query_one(
            f"SELECT MAX(queue_id) AS queue_id FROM {self._TABLE} WHERE agent_id = ?",
            (self._agent_id,),
        )
        return int(row["queue_id"]) if row else 0

    def pending(self) -> list[tuple[int, WriteResult]]:
        """Everything awaiting a curator, oldest first."""
        rows = self._store.iter_rows(
            f"SELECT queue_id, payload FROM {self._TABLE} "
            "WHERE agent_id = ? AND status = 'pending' ORDER BY queue_id",
            (self._agent_id,),
        )
        return [
            (int(row["queue_id"]), WriteResult.model_validate(json.loads(str(row["payload"]))))
            for row in rows
        ]

    def purge(self) -> int:
        """Remove every one of this agent's rows. Operator reset only.

        Deliberately outside the write pipeline: this is `make clean-memory`
        wiping a demo back to empty, not a memory operation with a
        principal — the same standing as rebuilding the database.
        """
        result = self._store.execute(
            f"DELETE FROM {self._TABLE} WHERE agent_id = ?", (self._agent_id,)
        )
        return result.affected

    def decide(self, queue_id: int, status: str, curator: str) -> None:
        self._store.execute(
            f"UPDATE {self._TABLE} SET status = ?, decided_by = ?, decided_at = ? "
            "WHERE agent_id = ? AND queue_id = ?",
            (status, curator, utc_now().isoformat(), self._agent_id, queue_id),
        )


class InMemoryCuratorQueue:
    """The same queue for a pipeline built without a platform store (tests)."""

    def __init__(self) -> None:
        self._items: dict[int, tuple[str, WriteResult]] = {}

    def put(self, result: WriteResult) -> int:
        queue_id = len(self._items) + 1
        self._items[queue_id] = ("pending", result)
        return queue_id

    def pending(self) -> list[tuple[int, WriteResult]]:
        return [(qid, item) for qid, (status, item) in self._items.items() if status == "pending"]

    def decide(self, queue_id: int, status: str, curator: str) -> None:
        if queue_id in self._items:
            self._items[queue_id] = (status, self._items[queue_id][1])
