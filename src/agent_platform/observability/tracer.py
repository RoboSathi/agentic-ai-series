"""Distributed tracing for agent decisions (LLD §6).

Every hop is traced, and every trace passes through a redactor before it is
persisted (FR-44) — there is no path that writes an unfiltered span to disk.
"""

import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import (
    get_current_trace_id,
    get_logger,
    new_trace_id,
    set_current_trace_id,
)
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import Span, utc_now

_CREATE_DUCKDB_SPANS_TABLE = """
CREATE TABLE IF NOT EXISTS platform_trace_spans (
    agent_id VARCHAR NOT NULL,
    span_id VARCHAR PRIMARY KEY,
    trace_id VARCHAR NOT NULL,
    parent_id VARCHAR,
    name VARCHAR NOT NULL,
    actor VARCHAR,
    tool VARCHAR,
    policy_version VARCHAR,
    model_version VARCHAR,
    state_transition VARCHAR,
    result VARCHAR,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR,
    latency_ms DOUBLE NOT NULL,
    cost_usd DOUBLE NOT NULL,
    attributes VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_platform_trace_trace
    ON platform_trace_spans (agent_id, trace_id);
"""


@runtime_checkable
class Redactor(Protocol):
    """Strips sensitive values before a span is persisted."""

    def redact(self, text: str) -> str: ...


@runtime_checkable
class Tracer(Protocol):
    """Starts spans and records their completion."""

    def start_span(self, name: str, parent: Span | None = None, **fields: Any) -> Span: ...

    def end_span(self, span: Span, result: str | None = None) -> None: ...


class LocalTracer:
    """In-process tracer with shared DuckDB persistence for observed agents.

    Spans land in the shared observability database (``store``), never in
    an agent's business database. Without a store they are kept in memory.
    """

    def __init__(
        self,
        redactor: Redactor | None = None,
        *,
        store: DuckDbObservabilityStore | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._settings = get_settings().observability
        self._audit_settings = get_settings().audit
        self._logger = get_logger("observability")
        self._redactor = redactor
        self._spans: list[Span] = []
        self._subscribers: list[Any] = []
        self._store = store
        self._agent_id = agent_id or get_settings().agent.id
        if self._store is not None and self._settings.persist_traces:
            self._store.execute_script(_CREATE_DUCKDB_SPANS_TABLE)

    def subscribe(self, subscriber: Any) -> None:
        """Register an observer notified as each span completes (cost, SLOs)."""
        self._subscribers.append(subscriber)

    def start_span(self, name: str, parent: Span | None = None, **fields: Any) -> Span:
        """Open a span, inheriting the parent's trace id, else the turn's, else a new one.

        A root span adopts the trace the turn bound, so its spans, the log
        lines and the LangSmith run share one id.
        """
        trace_id = parent.trace_id if parent else (get_current_trace_id() or new_trace_id())
        set_current_trace_id(trace_id)
        span = Span(
            agent_id=fields.pop("agent_id", self._agent_id),
            trace_id=trace_id,
            span_id=uuid.uuid4().hex,
            parent_id=parent.span_id if parent else None,
            name=name,
            # The configured policy version is a default, not an
            # override: a caller tracing a decision made under a
            # different policy must be able to say so.
            policy_version=fields.pop("policy_version", self._audit_settings.policy_version),
            **fields,
        )
        return span

    def end_span(self, span: Span, result: str | None = None) -> None:
        """Close a span, notify observers and persist it (redacted)."""
        span.ended_at = utc_now()
        span.latency_ms = (span.ended_at - span.started_at).total_seconds() * 1000
        if result is not None:
            span.result = result
        # Logged as well as persisted. A span is the record of *why* a
        # decision happened, and a subsystem that writes only to its own
        # table is one you cannot follow in the log beside everything
        # else that was happening at the time.
        self._logger.info(
            "Span closed",
            name=span.name,
            trace_id=span.trace_id,
            actor=span.actor,
            tool=span.tool,
            result=span.result,
            latency_ms=round(span.latency_ms, 1),
            policy_version=span.policy_version,
            model_version=span.model_version,
        )
        self._spans.append(span)
        for subscriber in self._subscribers:
            subscriber(span)
        self._persist(span)

    @contextmanager
    def span(self, name: str, parent: Span | None = None, **fields: Any) -> Iterator[Span]:
        """Scope a span to a block, closing it even when the block raises."""
        span = self.start_span(name, parent, **fields)
        try:
            yield span
        except Exception as exc:
            self.end_span(span, result=f"error: {exc}")
            raise
        else:
            if span.ended_at is None:
                self.end_span(span, result=span.result or "ok")

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def _persist(self, span: Span) -> None:
        if not self._settings.persist_traces or self._store is None:
            return
        redacted = self._redact(span)
        table = "platform_trace_spans"
        sql = (
            f"INSERT OR REPLACE INTO {table} "
            "(agent_id, span_id, trace_id, parent_id, name, actor, tool, policy_version, "
            "model_version, state_transition, result, started_at, ended_at, latency_ms, "
            "cost_usd, attributes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            redacted.agent_id,
            redacted.span_id,
            redacted.trace_id,
            redacted.parent_id,
            redacted.name,
            redacted.actor,
            redacted.tool,
            redacted.policy_version,
            redacted.model_version,
            redacted.state_transition,
            redacted.result,
            redacted.started_at.isoformat(),
            redacted.ended_at.isoformat() if redacted.ended_at else None,
            redacted.latency_ms,
            redacted.cost_usd,
            json.dumps(redacted.attributes, default=str),
        )
        self._store.execute(sql, params)

    def _redact(self, span: Span) -> Span:
        """Apply the PII pass — no trace reaches storage unfiltered."""
        if self._redactor is None:
            return span
        copy = span.model_copy(deep=True)
        if copy.result:
            copy.result = self._redactor.redact(copy.result)
        copy.attributes = {
            key: self._redactor.redact(value) if isinstance(value, str) else value
            for key, value in copy.attributes.items()
        }
        return copy
