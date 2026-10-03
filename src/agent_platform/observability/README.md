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
- `LocalTracer`, `LangSmithTracer`, `traced` (`tracer.py`, `tracing.py`)
- `DuckDbObservabilityStore` (`store.py`) — shared analytics file
- `CostAggregatorImpl` (`cost.py`), `SLORegistry`/`SLOTracker` (`slo.py`), `LittlesLawQueueModel` (`queue.py`), `PostmortemRegister` (`postmortem.py`)

## Configuration

Section `observability:` — `enabled` (tracing + SLOs), `analytics_db_path`, `langsmith_enabled`,
`persist_traces`, `slo_targets`, `review_queue`.

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
