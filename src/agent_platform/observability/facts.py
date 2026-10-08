"""Analytic facts: one row per model call, tool call and turn (LLD §6).

Spans say *why* a decision happened; facts say *how much* and *how long*, in a
shape aggregate questions can be asked of (p95 latency by model, tokens per
answered turn, tool failure rate this week). Each fact is a typed model whose
table is derived from its fields, so adding a column is adding a field — there
is no second schema to keep in step.

A fact is stamped with the agent, session, trace and channel bound by the turn
(`logging.bound_log_context`), so the recording component never has to know
which turn it is in. Recording never raises: a turn must not fail because its
bookkeeping did. A field left `None` means "not observed by this agent", which
is different from zero and is stored as NULL, so an SLO is computed only over
the agents that can observe it.
"""

from __future__ import annotations

import types
from datetime import datetime
from typing import Any, ClassVar, Protocol, Union, get_args, get_origin, runtime_checkable

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.logging.trace_context import log_context_fields
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import utc_now

# Python type → DuckDB column type, for the fields a fact may carry.
_COLUMN_TYPES: dict[type, str] = {
    bool: "BOOLEAN",
    int: "BIGINT",
    float: "DOUBLE",
    str: "VARCHAR",
    datetime: "TIMESTAMPTZ",
}
# The log-context name each identity field is stamped from.
_CONTEXT_FIELDS = {
    "agent_id": "agent",
    "session_id": "session",
    "trace_id": "trace_id",
    "channel": "channel",
}


class Fact(BaseModel):
    """What every fact carries: whose, which conversation, which turn, when."""

    table: ClassVar[str] = ""

    agent_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    channel: str = ""
    recorded_at: datetime = Field(default_factory=utc_now)


class LlmCallFact(Fact):
    """One provider call: one attempt on one model, as the adapter saw it."""

    table: ClassVar[str] = "platform_llm_calls"

    role: str = ""
    provider: str = ""
    model: str = ""
    # 0 is the role's primary model; 1.. are its backups, in chain order.
    position: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    succeeded: bool = True
    error: str = ""


# A tool call's status outside the Executor, in the Executor's own words.
TOOL_COMPLETED, TOOL_FAILED = "completed", "failed"


class ToolCallFact(Fact):
    """One tool execution, after authorization."""

    table: ClassVar[str] = "platform_tool_calls"

    action: str = ""
    actor: str = ""
    status: str = ""
    succeeded: bool = True
    side_effecting: bool = False
    replayed: bool = False
    timed_out: bool = False
    duration_ms: float = 0.0
    error: str = ""


class TurnOutcomeFact(Fact):
    """How one turn went — the observations every SLO is computed from."""

    table: ClassVar[str] = "platform_turn_outcomes"

    outcome: str = ""
    latency_ms: float = 0.0
    answered: bool | None = None
    escalated: bool | None = None
    unauthorized_attempts: int | None = None
    policy_refusals: int | None = None
    loop_aborted: bool | None = None
    hallucination_rate: float | None = None
    tool_calls: int | None = None
    tool_failures: int | None = None
    recoveries_attempted: int | None = None
    recoveries_succeeded: int | None = None
    # Claims in the reply its grounding check could not support (kept, flagged or removed).
    unverified_claims: int | None = None


@runtime_checkable
class FactRecorder(Protocol):
    """Where a component hands the facts it observes."""

    def record(self, fact: Fact) -> None: ...


class NullFactRecorder:
    """Records nothing: a component built without the fact ledger."""

    def record(self, fact: Fact) -> None:
        return None


class DuckDbFactLedger:
    """Facts in the shared observability database, one table per fact type."""

    def __init__(self, store: DuckDbObservabilityStore, agent_id: str | None = None) -> None:
        self._store = store
        self._agent_id = agent_id or get_settings().agent.id
        self._tables: set[str] = set()
        self._logger = get_logger("observability")

    def record(self, fact: Fact) -> None:
        """Stamp the fact with the turn it happened in and append it."""
        stamped = self._stamped(fact)
        row = stamped.model_dump()
        columns = list(row)
        sql = (
            f"INSERT INTO {stamped.table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )
        try:
            self._ensure_table(type(stamped))
            self._store.execute(sql, tuple(row.values()))
        except Exception as exc:  # noqa: BLE001 - bookkeeping must never fail a turn
            self._logger.warning("Fact not recorded", table=stamped.table, error=str(exc))

    def _stamped(self, fact: Fact) -> Fact:
        """Fill whatever identity the recorder left blank from the bound turn context."""
        bound = log_context_fields()
        update = {
            field: bound[name]
            for field, name in _CONTEXT_FIELDS.items()
            if not getattr(fact, field) and name in bound
        }
        if not (update.get("agent_id") or fact.agent_id):
            update["agent_id"] = self._agent_id
        return fact.model_copy(update=update)

    def _ensure_table(self, model: type[Fact]) -> None:
        if model.table in self._tables:
            return
        self._store.execute_script(fact_table_ddl(model))
        self._tables.add(model.table)


def fact_table_ddl(model: type[Fact]) -> str:
    """The table a fact type implies, one column per field — created, or brought up to date.

    Evolution is additive: a field added to the model becomes a column added to an
    existing table (NULL in the rows written before it), so a database from an
    earlier version keeps working without a migration step.
    """
    columns = {name: _column_type(info.annotation) for name, info in model.model_fields.items()}
    created = ",\n    ".join(f"{name} {kind}" for name, kind in columns.items())
    added = "".join(
        f"ALTER TABLE {model.table} ADD COLUMN IF NOT EXISTS {name} {kind};\n"
        for name, kind in columns.items()
    )
    return (
        f"CREATE TABLE IF NOT EXISTS {model.table} (\n    {created}\n);\n{added}"
        f"CREATE INDEX IF NOT EXISTS idx_{model.table}_trace ON {model.table} (agent_id, trace_id);"
    )


def _column_type(annotation: Any) -> str:
    """The column type for a field, looking through `X | None`."""
    if get_origin(annotation) in (Union, types.UnionType):
        annotation = next(arg for arg in get_args(annotation) if arg is not type(None))
    return _COLUMN_TYPES[annotation]


def create_fact_recorder(store: DuckDbObservabilityStore | None, agent_id: str) -> FactRecorder:
    """The configured recorder: the DuckDB ledger when facts are persisted, else nothing."""
    if store is None or not get_settings().observability.persist_facts:
        return NullFactRecorder()
    return DuckDbFactLedger(store, agent_id=agent_id)


__all__ = [
    "DuckDbFactLedger",
    "Fact",
    "FactRecorder",
    "LlmCallFact",
    "NullFactRecorder",
    "TOOL_COMPLETED",
    "TOOL_FAILED",
    "ToolCallFact",
    "TurnOutcomeFact",
    "create_fact_recorder",
    "fact_table_ddl",
]
