# ADR 0007 — Analytic facts, and a Parquet mart every reader shares

- Status: accepted
- Date: 2026-10-08

## Context

Two features were asked for: natural-language questions over the platform's
observability data (through an MCP server), and dashboards over it. Looking at
the data first showed three problems:

- **It was nearly empty.** All 2,832 spans in `observability.duckdb` were CSA
  `llm_call` spans with `latency_ms = 0`, `cost_usd = 0` and no attributes: the
  span was opened and closed *after* the call. The research agent wrote none.
  Tokens were metered for the status bar and never persisted. Only the audit
  trail held real signal.
- **SLOs could not be analysed.** `SLORegistry` keeps an in-memory list whose
  mean covers the whole process lifetime: no window, so no error budget, and
  nothing after a restart. `reason.py` also recorded `tool_failure_rate = 0.0`
  on every model call, a control state no failure could move.
- **DuckDB has one writer.** The running console holds the lock on
  `observability.duckdb`; a separate MCP server or Rill cannot open it.

## Decision

1. **Facts, not spans, are the analytic record.** `observability/facts.py` adds
   typed facts — `LlmCallFact` (every provider attempt, recorded by a
   `RecordedClient` decorator that `ModelChainBuilder` puts inside each model's
   resilience stack), `ToolCallFact` (recorded by the Executor, always on) and
   `TurnOutcomeFact` (recorded by each agent at the end of a turn). The table is
   derived from the model's fields; identity (agent, session, trace, channel) is
   stamped from the bound log context; a field an agent does not observe is
   NULL, not zero; recording never raises.
2. **SLO observations are persisted as turn facts; SLOs are derived.** The mart
   computes daily SLOs and windowed error budgets (`analytics.slo_windows_days`)
   from `platform_turn_outcomes`, with the registry's `LOWER_IS_BETTER` rule. The
   in-memory registry stays as the live per-process view.
3. **Readers share a Parquet mart.** `analytics/semantic_layer.yaml` is the one
   place each curated view's SQL and column meanings live. `MartExporter` renders
   it on the process holding the lock (a running console, every
   `analytics.export_interval_seconds`, or `make analytics-export`), writes each
   view to `data/analytics/<view>.parquet` by atomic rename, and fails the export
   if a view's columns drift from their documentation. The audit view extracts
   standing, session and authority source from the JSON and never exports the raw
   `authority_context` (it carries unredacted request data); free text is masked
   by `analytics.redact_patterns`.
4. **The CSA `llm_call` span now times the real call**, and the bogus SLO
   observation and the dead `cost_per_success_usd` entry are removed.

## Consequences

- The MCP server and Rill read the same views with the same meanings.
- A dashboard is at most one export interval stale while an agent runs.
- Cost stays tokens: there is still no rate card (`accounting.py`).
- Audit hygiene at the source (standing/session columns, curated and redacted
  `authority_context`, RA `model_version`, settlement causation) is a separate
  track; the mart works around it.
- `/audit` read an empty SQLite ledger while every write went to DuckDB; it now
  reads `DuckDbAuditLedger.recent()`. `/dlq` and `make dlq` still write their
  audit entries to SQLite (an out-of-process writer would hit the lock) — owed.

## Addendum (2026-10-08) — two dashboards, not one per table

Six explores, one per raw view, answered nothing on their own, and four of them
failed to load while their tables were empty. They were replaced by two, each
over a purpose-built view:

- **Agent Health** over a widened `turns` (each turn joined to its model calls,
  tool calls and authority decisions by `(agent_id, trace_id)`): the SLO view.
- **Calls & Decisions** over `events` (model calls, tool calls and decisions as
  one event stream): the "which dependency or rule" view, holding the audit history.

For the join, research-agent audit entries now carry the turn's trace id (the run
id moves to `authority_context.run_id`), `TurnOutcomeFact` gains
`unverified_claims`, and fact tables evolve additively (`ADD COLUMN IF NOT
EXISTS`), so an older database gains new columns without a migration.
