# observability — traces, costs, SLOs and the live feed

Everything the harness reports about itself while it runs: the live
status feed the TUI renders, spans (local and LangSmith), cost per stage,
SLO targets and error budgets, queue models for human review, postmortems,
and a shared DuckDB analytics store for traces and audit data.

## Vocabulary

- **StatusEvent** — one stage starting, finishing or failing — what the console shows
- **StageReporter** — how a component emits status events
- **SLO** — a target (task success, hallucination rate, …) with an error budget

## Public interface

- `StageReporter`, `InMemoryLiveStatusFeed`, `NullLiveStatusFeed`, `StatusEvent` (`status_feed.py`, `types.py`)
- Card fields every agent shares: `llm_request_fields`/`llm_response_fields` (`llm_fields.py`), `tool_request_fields`/`tool_result_fields`/`control_loop_fields` (`tool_fields.py`)
- `LoopReporter`, `mark_step` (`loop_reporter.py`): a reasoning loop's cards numbered "Step N", with the cap beside it
- `LocalTracer`, `traced` (`tracer.py`, `tracing.py`); `LangSmithExporter`, `turn_trace` (`langsmith.py`)
- `DuckDbObservabilityStore` (`store.py`) — shared analytics file
- Analytics facts (ADR 0009): `FactStore`, `install_fact_store`, the Tier-1 fact models (`facts.py`);
  the turn bracket and task API `open_task`/`close_task`/`mark_cancelled`/`end_session`,
  `OutcomeCatalog` (`units.py`, context in `unit_context.py`); the check recorder
  `record_check`/`record_screen`/`recorded_dependency` (`checks.py`)
- `CostAggregatorImpl` (`cost.py`), `SLORegistry`/`SLOTracker` (`slo.py`), `LittlesLawQueueModel` (`queue.py`), `PostmortemRegister` (`postmortem.py`)

## Configuration

Section `observability:` — `enabled` (tracing + SLOs), `analytics_db_path`, `langsmith`
(per-turn trace export, `langsmith.py`), `persist_traces`, `slo_targets`, `review_queue`.

## Seams

- The feed and reporter are injected into the agent (`build_agent(reporter=...)`).

## Collaborators

- Consumed by `tui`; the audit ledger can share its DuckDB store. Eval analytics use their own DuckDB file (`evals.store`) so a running console never locks an eval out.

## Episodes

E01, E08, E16, E36–E38, E44, E46, E48c.

## Tests

Run with `./venv/bin/pytest tests/unit/test_observability.py -q` (or `make test` for everything).

## Status and limits

`PostmortemRegister` has no console command any more (`/postmortem` was removed); it remains public API with its tests.
