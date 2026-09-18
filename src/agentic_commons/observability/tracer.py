"""Distributed tracing for agent decisions (E36, LLD §6).

Every hop is traced, and every trace passes through a redactor before it is
persisted (FR-44) — there is no path that writes an unfiltered span to disk.
"""

import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, set_current_trace_id
from agentic_commons.observability.types import Span, utc_now
from agentic_commons.persistence import SqliteConnectionPool

_CREATE_SPANS_TABLE = """
CREATE TABLE IF NOT EXISTS trace_spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    actor TEXT,
    tool TEXT,
    policy_version TEXT,
    model_version TEXT,
    state_transition TEXT,
    result TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    attributes TEXT NOT NULL
)
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
    """In-process tracer that optionally persists spans to SQLite.

    Spans are kept in memory for the current process (so a test or the TUI
    can reconstruct a trace immediately) and written through the redactor
    when persistence is enabled.
    """

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._settings = get_settings().observability
        self._audit_settings = get_settings().audit
        self._logger = get_logger("observability")
        self._redactor = redactor
        self._spans: list[Span] = []
        self._subscribers: list[Any] = []
        self._pool = pool
        if self._pool is not None and self._settings.persist_traces:
            self._pool.execute_script(_CREATE_SPANS_TABLE)

    def subscribe(self, subscriber: Any) -> None:
        """Register an observer notified as each span completes (cost, SLOs)."""
        self._subscribers.append(subscriber)

    def start_span(self, name: str, parent: Span | None = None, **fields: Any) -> Span:
        """Open a span, inheriting the parent's trace id when there is one."""
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        set_current_trace_id(trace_id)
        span = Span(
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

    def spans_for(self, trace_id: str) -> list[Span]:
        """Every span of one trace, in causal (start-time) order."""
        return sorted(
            [span for span in self._spans if span.trace_id == trace_id],
            key=lambda span: span.started_at,
        )

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def _persist(self, span: Span) -> None:
        if self._pool is None or not self._settings.persist_traces:
            return
        redacted = self._redact(span)
        self._pool.execute(
            "INSERT OR REPLACE INTO trace_spans (span_id, trace_id, parent_id, name, actor, "
            "tool, policy_version, model_version, state_transition, result, started_at, "
            "ended_at, latency_ms, cost_usd, attributes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
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
                str(redacted.attributes),
            ),
        )

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


class LangSmithTracer:
    """Dual-write wrapper that also reports spans to LangSmith.

    LangSmith is an optional external dependency: if it is disabled or the
    client can't be constructed, this degrades to the wrapped tracer rather
    than failing the request.
    """

    def __init__(self, inner: LocalTracer) -> None:
        self._inner = inner
        self._logger = get_logger("observability")
        self._client = self._build_client()

    def _build_client(self) -> Any:
        if not get_settings().observability.langsmith_enabled:
            return None
        try:
            from langsmith import Client

            return Client()
        except Exception as exc:
            self._logger.warning("LangSmith unavailable; local tracing only", error=str(exc))
            return None

    def start_span(self, name: str, parent: Span | None = None, **fields: Any) -> Span:
        return self._inner.start_span(name, parent, **fields)

    def end_span(self, span: Span, result: str | None = None) -> None:
        self._inner.end_span(span, result)
        if self._client is None:
            return
        try:
            self._client.create_run(
                name=span.name,
                run_type="chain",
                inputs={"actor": span.actor, "tool": span.tool},
                outputs={"result": span.result},
                extra={"trace_id": span.trace_id, "cost_usd": span.cost_usd},
            )
        except Exception as exc:
            self._logger.warning("LangSmith span export failed", error=str(exc))

    def spans_for(self, trace_id: str) -> list[Span]:
        return self._inner.spans_for(trace_id)
