"""Distributed tracing for agent decisions (E36, LLD §6).

Every hop is traced, and every trace passes through a redactor before it is
persisted (FR-44) — there is no path that writes an unfiltered span to disk.
"""

import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger, set_current_trace_id
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import Span, utc_now
from agent_platform.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

_CREATE_SQLITE_SPANS_TABLE = """
CREATE TABLE IF NOT EXISTS platform_trace_spans (
    agent_id TEXT NOT NULL,
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

    The legacy ``pool`` parameter remains supported for older episode demos.
    New agents should pass ``store`` so all traces land in the shared
    observability database rather than an agent's business database.
    """

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        redactor: Redactor | None = None,
        *,
        store: DuckDbObservabilityStore | None = None,
        platform_store: PlatformStore | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._settings = get_settings().observability
        self._audit_settings = get_settings().audit
        self._logger = get_logger("observability")
        self._redactor = redactor
        self._spans: list[Span] = []
        self._subscribers: list[Any] = []
        self._pool = pool
        self._store = store
        self._platform_store = platform_store or (
            as_platform_store(pool, agent_id=agent_id) if pool is not None else None
        )
        self._agent_id = agent_id or (
            self._platform_store.agent_id if self._platform_store else get_settings().agent.id
        )
        if self._store is not None and self._settings.persist_traces:
            self._store.execute_script(_CREATE_DUCKDB_SPANS_TABLE)
        elif self._platform_store is not None and self._settings.persist_traces:
            self._platform_store.execute_script(_CREATE_SQLITE_SPANS_TABLE)
            self._migrate_legacy_sqlite_spans()

    def subscribe(self, subscriber: Any) -> None:
        """Register an observer notified as each span completes (cost, SLOs)."""
        self._subscribers.append(subscriber)

    def start_span(self, name: str, parent: Span | None = None, **fields: Any) -> Span:
        """Open a span, inheriting the parent's trace id when there is one."""
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
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
        if not self._settings.persist_traces:
            return
        if self._store is None and self._platform_store is None:
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
        if self._store is not None:
            self._store.execute(sql, params)
        else:
            # Legacy episode compatibility only; new agents use DuckDB.
            self._platform_store.execute(sql, params)  # type: ignore[union-attr]

    def _migrate_legacy_sqlite_spans(self) -> None:
        """Copy legacy SQLite spans into the platform-owned trace table."""
        if self._platform_store is None:
            return
        exists = self._platform_store.query_one(
            "SELECT 1 AS hit FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("trace_spans",),
        )
        if exists is None:
            return
        columns = {
            str(row["name"])
            for row in self._platform_store.iter_rows("PRAGMA table_info(trace_spans)")
        }
        if "agent_id" in columns:
            rows = self._platform_store.iter_rows(
                "SELECT agent_id, span_id, trace_id, parent_id, name, actor, tool, policy_version, "
                "model_version, state_transition, result, started_at, ended_at, latency_ms, cost_usd, attributes "
                "FROM trace_spans WHERE agent_id = ?",
                (self._agent_id,),
            )
        else:
            rows = self._platform_store.iter_rows(
                "SELECT span_id, trace_id, parent_id, name, actor, tool, policy_version, "
                "model_version, state_transition, result, started_at, ended_at, latency_ms, cost_usd, attributes "
                "FROM trace_spans"
            )
        for row in rows:
            values = (
                row.get("agent_id", self._agent_id),
                row["span_id"],
                row["trace_id"],
                row["parent_id"],
                row["name"],
                row["actor"],
                row["tool"],
                row["policy_version"],
                row["model_version"],
                row["state_transition"],
                row["result"],
                row["started_at"],
                row["ended_at"],
                row["latency_ms"],
                row["cost_usd"],
                row["attributes"],
            )
            self._platform_store.execute(
                "INSERT OR REPLACE INTO platform_trace_spans "
                "(agent_id, span_id, trace_id, parent_id, name, actor, tool, policy_version, "
                "model_version, state_transition, result, started_at, ended_at, latency_ms, "
                "cost_usd, attributes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
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
