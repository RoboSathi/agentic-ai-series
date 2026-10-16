"""Analytic facts: the Tier-1 fact tables and the store every fact is appended through.

ADR 0009 and `design_docs/analytics_schema.md` §2. One row per task event, turn,
model-call attempt, tool proposal and check, each written by the one platform
choke point that observes it. A fact is a typed model whose table is derived
from its fields, so adding a column is adding a field — there is no second
schema to keep in step, and evolution is additive.

Every fact carries the common keys (D5), stamped by the store from the bound
context — agent, session and channel from the log context, turn, task and step
from the unit context — so the recording component never has to know which
turn it is in. Appending never raises: a turn must not fail because its
bookkeeping did. A field left `None` means "not observed" (the agent withholds
the feature, or has no such stage), which is different from zero and is stored
as NULL.

Agents add Tier-2 facts the same way: a `Fact` subclass whose table is
prefixed with the agent's id (D7), appended through the same store.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar, Protocol, Union, get_args, get_origin, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.logging.trace_context import log_context_fields
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import utc_now
from agent_platform.observability.unit_context import unit_fields

# Python type → DuckDB column type, for the fields a fact may carry.
_COLUMN_TYPES: dict[Any, str] = {
    bool: "BOOLEAN",
    int: "BIGINT",
    float: "DOUBLE",
    str: "VARCHAR",
    datetime: "TIMESTAMPTZ",
    list[str]: "VARCHAR[]",
}
# The log-context name each identity key is stamped from.
_CONTEXT_FIELDS = {"agent_id": "agent", "session_id": "session", "channel": "channel"}


class Fact(BaseModel):
    """The common keys (D5): whose, which conversation, task, turn and step, where, when."""

    table: ClassVar[str] = ""

    agent_id: str = ""
    session_id: str = ""
    task_id: str = ""
    turn_id: str = ""
    step: int | None = None
    channel: str = ""
    recorded_at: datetime = Field(default_factory=utc_now)


class TaskEventFact(Fact):
    """A task opened or closed (`units.open_task` / `close_task`, or the bracket)."""

    table: ClassVar[str] = "fact_task_event"

    event: str = ""
    kind: str = ""
    outcome: str | None = None
    success: bool | None = None
    terminal: bool | None = None


class TurnFact(Fact):
    """One turn, written by the bracket when it ends, whatever ended it."""

    table: ClassVar[str] = "fact_turn"

    status: str = ""
    latency_ms: float = 0.0
    steps: int = 0
    step_cap: int | None = None
    step_cap_hit: bool | None = None
    claims_checked: int | None = None
    claims_supported: int | None = None
    claims_unsupported: int | None = None
    # The screen checks this agent's composition root built (None: no screen at all).
    screens: list[str] | None = None
    error: str = ""


class ModelCallFact(Fact):
    """One attempt on one model, as the adapter saw it."""

    table: ClassVar[str] = "fact_model_call"

    role: str = ""
    provider: str = ""
    model: str = ""
    attempt: int = 1
    is_backup: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    succeeded: bool = True
    error: str = ""


class ToolCallFact(Fact):
    """One tool proposal, at whichever stage of the single tool path it stopped."""

    table: ClassVar[str] = "fact_tool_call"

    tool: str = ""
    stage: str = ""
    succeeded: bool | None = None
    timed_out: bool | None = None
    duration_ms: float | None = None
    side_effecting: bool | None = None
    reason: str = ""


class CheckFact(Fact):
    """One invocation of a guard, classifier or model dependency on content."""

    table: ClassVar[str] = "fact_check"

    check_name: str = ""
    purpose: str = ""
    backend: str = ""
    model: str = ""
    verdict: str = ""
    score: float | None = None
    latency_ms: float = 0.0
    items: int | None = None
    flagged: int | None = None


# A task event (`TaskEventFact.event`): platform vocabulary, the same for every agent.
OPENED, CLOSED = "opened", "closed"

# The Tier-1 facts, in the order the schema documents them.
TIER1_FACTS: tuple[type[Fact], ...] = (
    TaskEventFact,
    TurnFact,
    ModelCallFact,
    ToolCallFact,
    CheckFact,
)


@runtime_checkable
class FactStore(Protocol):
    """Where every fact is appended, and the one read the turn bracket needs."""

    def append(self, fact: Fact) -> None: ...

    def open_task(self, agent_id: str, session_id: str) -> tuple[str, str] | None:
        """The session's open explicit task as (task_id, kind), if one was left open."""
        ...


class NullFactStore:
    """Keeps nothing: an agent built without persisted facts."""

    def append(self, fact: Fact) -> None:
        return None

    def open_task(self, agent_id: str, session_id: str) -> tuple[str, str] | None:
        return None


class DuckDbFactStore:
    """Facts in the shared observability database, one table per fact type."""

    def __init__(self, store: DuckDbObservabilityStore, agent_id: str | None = None) -> None:
        self._app_config = get_settings()
        self._store = store
        self._agent_id = agent_id or get_settings().agent.id
        self._tables: set[str] = set()
        self._logger = get_logger("observability")

    def append(self, fact: Fact) -> None:
        """Stamp the fact with the units it happened in and append it; never raise."""
        try:
            stamped = self._stamped(fact)
            row = stamped.model_dump()
            self._ensure_table(type(stamped))
            self._store.execute(_insert_sql(stamped.table, list(row)), tuple(row.values()))
        except Exception as exc:  # noqa: BLE001 - bookkeeping must never fail a turn
            self._logger.warning("Fact not recorded", table=fact.table, error=str(exc))

    def append_many(self, facts: Sequence[Fact]) -> None:
        """Append facts in one batch per table (a backfill, a generated world); never raise."""
        by_table: dict[type[Fact], list[Fact]] = {}
        for fact in facts:
            by_table.setdefault(type(fact), []).append(fact)
        for model, rows in by_table.items():
            try:
                self._ensure_table(model)
                columns = {
                    name: _column_type(f.annotation) for name, f in model.model_fields.items()
                }
                self._store.bulk_insert(
                    model.table, columns, [self._stamped(fact).model_dump() for fact in rows]
                )
            except Exception as exc:  # noqa: BLE001 - bookkeeping must never fail a caller
                self._logger.warning("Facts not recorded", table=model.table, error=str(exc))

    def open_task(self, agent_id: str, session_id: str) -> tuple[str, str] | None:
        """The latest explicit task opened in this session and never closed."""
        try:
            self._ensure_table(TaskEventFact)
            row = self._store.query_one(_OPEN_TASK_SQL, (agent_id, session_id, _implicit_kind()))
        except Exception as exc:  # noqa: BLE001 - a lookup miss only means a fresh task
            self._logger.warning("Open task not looked up", error=str(exc))
            return None
        return (str(row["task_id"]), str(row["kind"])) if row else None

    def _stamped(self, fact: Fact) -> Fact:
        """Fill whatever common key the recorder left blank from the bound context."""
        bound = log_context_fields()
        update: dict[str, Any] = {
            key: bound[name]
            for key, name in _CONTEXT_FIELDS.items()
            if not getattr(fact, key) and name in bound
        }
        update.update(
            {key: value for key, value in unit_fields().items() if getattr(fact, key) in ("", None)}
        )
        if not (update.get("agent_id") or fact.agent_id):
            update["agent_id"] = self._agent_id
        if not (update.get("channel") or fact.channel):
            update["channel"] = self._app_config.analytics.default_channel
        return fact.model_copy(update=update)

    def _ensure_table(self, model: type[Fact]) -> None:
        if model.table in self._tables:
            return
        self._store.execute_script(fact_table_ddl(model))
        self._tables.add(model.table)


def _implicit_kind() -> str:
    """Read at call time, so an overlay can rename the implicit kind."""
    return get_settings().analytics.tasks.implicit_kind


_OPEN_TASK_SQL = f"""
SELECT o.task_id, o.kind FROM {TaskEventFact.table} o
WHERE o.event = '{OPENED}' AND o.agent_id = ? AND o.session_id = ? AND o.kind <> ?
  AND NOT EXISTS (
    SELECT 1 FROM {TaskEventFact.table} c WHERE c.task_id = o.task_id AND c.event = '{CLOSED}'
  )
ORDER BY o.recorded_at DESC LIMIT 1
"""


def _insert_sql(table: str, columns: list[str]) -> str:
    quoted = ", ".join(f'"{name}"' for name in columns)
    return f"INSERT INTO {table} ({quoted}) VALUES ({', '.join('?' for _ in columns)})"


def fact_table_ddl(model: type[Fact]) -> str:
    """The table a fact type implies, one column per field — created, or brought up to date.

    Evolution is additive: a field added to the model becomes a column added to an
    existing table (NULL in the rows written before it), so a database from an
    earlier version keeps working without a migration step.
    """
    columns = {name: _column_type(info.annotation) for name, info in model.model_fields.items()}
    created = ",\n    ".join(f'"{name}" {kind}' for name, kind in columns.items())
    added = "".join(
        f'ALTER TABLE {model.table} ADD COLUMN IF NOT EXISTS "{name}" {kind};\n'
        for name, kind in columns.items()
    )
    return (
        f"CREATE TABLE IF NOT EXISTS {model.table} (\n    {created}\n);\n{added}"
        f"CREATE INDEX IF NOT EXISTS idx_{model.table}_task ON {model.table} (agent_id, task_id);"
    )


def _column_type(annotation: Any) -> str:
    """The column type for a field, looking through `X | None`."""
    if get_origin(annotation) in (Union, types.UnionType):
        annotation = next(arg for arg in get_args(annotation) if arg is not type(None))
    return _COLUMN_TYPES[annotation]


# The process's store: an agent process has one writer (schema §1), installed by its
# composition root. Components built with no store injected append here.
_installed: list[FactStore] = [NullFactStore()]


def install_fact_store(store: FactStore) -> None:
    """Make `store` the process's fact store (the composition root calls this once)."""
    _installed[0] = store


def fact_store() -> FactStore:
    """The process's fact store (a null store until a composition root installs one)."""
    return _installed[0]


def create_fact_store(store: DuckDbObservabilityStore | None, agent_id: str) -> FactStore:
    """The configured store: DuckDB when facts are persisted, else one that keeps nothing."""
    if store is None or not get_settings().observability.persist_facts:
        return NullFactStore()
    return DuckDbFactStore(store, agent_id=agent_id)


__all__ = [
    "CLOSED",
    "CheckFact",
    "DuckDbFactStore",
    "Fact",
    "FactStore",
    "ModelCallFact",
    "NullFactStore",
    "OPENED",
    "TIER1_FACTS",
    "TaskEventFact",
    "ToolCallFact",
    "TurnFact",
    "create_fact_store",
    "fact_store",
    "fact_table_ddl",
    "install_fact_store",
]
