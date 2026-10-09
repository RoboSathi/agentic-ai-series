# ADR 0009 — Analytics is a platform contract

- Status: accepted; implemented 2026-10-09 (see Implementation notes)
- Date: 2026-10-08
- Supersedes: the fact tables, mart views and dashboards of ADR 0007 (the
  Parquet mart, the exporter, the MCP server and the client of ADR 0008 stand)

## Context

The first analytics build (ADR 0007) grew from what was easy to log:

- **Turn facts were filled by each agent by hand.** A field one agent forgot was
  NULL, two agents meant different things by the same field, and CSA derived
  refusals and tool failures by matching strings in `state.decisions`.
- **Grains were mixed.** A research run spans many turns and had no row of its own.
  A tool call the model proposed but the platform rejected was not recorded at all.
- **Only model calls were recorded at a platform choke point.** Security screens,
  grounding and tool validity were drawn as cards or logged, but never kept.
- **Evals had their own schema,** sharing no keys with runtime data.

Dashboards built on that showed numbers nobody could act on. Analytics has to be
designed from the vocabulary down, and it has to hold for *any* agent built on
the platform, not just the three that exist.

## Decision

**Every agent built on `agent_platform` gets the common metrics without writing
analytics code.** A common metric is recorded by a platform choke point that the
agent cannot bypass. An agent contributes only three things: the declaration of
its outcomes, optionally its own domain tables, and its eval suite.

| # | Decision |
|---|---|
| D1 | **Task and Turn are both first-class.** A task is one unit of work, possibly spanning turns, opened and closed through a platform task API. A turn is one message in and one reply out, bracketed by the platform. |
| D2 | **Outcomes are agent-defined,** declared in the agent's config, in its own words. |
| D4 | **Outcome flags.** For each outcome it names, an agent declares `success` and `terminal`. The fleet resolution rate is successful ÷ terminal tasks. The outcome mix is shown only on the agent's own page. An undeclared outcome is a start-up error. |
| D5 | **Common keys on every fact:** `agent_id, session_id, task_id, turn_id, step, channel, recorded_at` (UTC). There is no release version or configuration fingerprint, so comparing releases is out of scope. Eval traffic is told apart by `channel = 'eval'`. |
| D6 | **The Tier-1 catalogue has 16 metrics** (see `analytics_schema.md`). Each is recorded only by a platform choke point. Where an agent withholds the feature a metric needs, the value is NULL, not 0. |
| D3 | **Dashboards:** one Fleet page (Tier 1, every agent side by side, generated from the metric catalogue) and one page per agent (its domain tables and its eval scorecard). |
| D7 | **Tier 2 lives in agent-owned tables.** An agent defines its own fact tables and views in its own package, and its overlay registers its semantic-layer file. The platform discovers registered layers and never names an agent's tables. |
| D8 | **Tier 3 evals have a common core plus agent metrics.** Every suite reports task success on its golden set, groundedness, tool-selection accuracy and attack success rate, plus its own metrics. Results go into one `eval_score` fact. The suites themselves are designed in the evals rework. |
| D9 | **Baseline refactors come first:** a single tool path; a turn bracket that records timing, crashes and steps; a task API; and check recording. The ad-hoc facts are dropped, not migrated. |

## Consequences

- A new agent appears on the Fleet page by running turns, with no analytics code. It gets a
  page of its own only if it registers domain tables or an eval suite.
- RA's direct-answer tools must stop bypassing the Executor, and every agent's tool
  calls go through one platform path. That path is what makes tool validity
  measurable, and it is also a governance fix.
- Every agent must open and close tasks. An agent that never closes one shows
  open tasks, not a fabricated outcome.
- The existing `platform_llm_calls` / `platform_tool_calls` /
  `platform_turn_outcomes` tables, the `turns` / `events` views and the two Rill
  dashboards are retired once the new facts exist.
- Releases cannot be compared (D5); if needed later, a fingerprint column is an additive change.
- Human review outcomes, intent classification and Gatekeeper standings are not
  Tier 1: they belong to the agents that have them (Tier 2), or come later.

## Implementation notes (2026-10-09)

Built as specified in `analytics_schema.md`, which has been updated in place. Where
the build differs from the design, these are the reasons:

- **`check` is `check_name`.** `check` is a reserved word in DuckDB, so every query
  over it, whether hand-written or model-written, would need quoting.
- **`fact_turn.screens`** (added). This is the list of screens the composition root
  declared (`units.declare_screens`). `screen_coverage` needs the declared set,
  because a screen that never ran leaves no row to infer it from. A turn that carries
  no message from a person (a resumed or aborted run) has none.
- **"Offered" stays with the Gatekeeper.** It is a fact about the instruction, which
  CLAUDE.md assigns to the Gatekeeper. `ToolPath.run(proposal, turn)` records
  `not_offered` when the Gatekeeper rules on its `offered` question, and with the
  Gatekeeper withheld, nothing checks it. `ToolTurn` carries the caller's authority
  builder, control loop, step budget and card hooks.
- **Agent layers are found by convention** (`analytics.agent_layers_glob`), not
  registered per overlay. The source holds every agent, whichever process exports it;
  per-overlay registration would make it depend on which agent last ran. Each
  layer names its `agent_id`. An agent view whose tables do not exist yet is skipped
  with its reason, together with the metrics over it.
- **Metrics have one form,** `{view, expr, by}`, rendered per day (`metric_daily`), per
  SLO window (`metric_window`) and per period (`compute_metric`). Two helper shapes make
  that possible: `tasks` carries tokens per task, and the `screen_coverage` view expands
  each turn's declared screens.
- **Reserved outcomes.** `unclassified` (not success, not terminal), and `crashed`,
  `cancelled`, `abandoned` (not success, terminal). An implicit task the agent does not
  close is `unclassified`, so it never counts toward resolution.
- **Steps** are counted by `SteppedClient`, the outermost wrapper the chain builder
  applies. A fact written outside a model call takes the turn's latest step: a tool
  proposal belongs to the step that proposed it, and a screen before the first step
  has none. The step *cap* is declared by the loop guard built for the turn
  (`ToolControlLoop(step_cap=...)`, or `BudgetGuard(step_counter=...)`).
- **Eval traffic** is bound to `channel = eval` (`evals.channel`). The CSA eval runner
  still builds with observability off, because the console holds the DuckDB lock, so
  eval facts arrive with the evals rework (D8), together with `eval_score`.
- **MCP:** `list_metrics` and `compute_metric(name, agent_id, days, by)` join the three
  view tools. A computed metric comes back in the query result's shape, so a client's
  figure check reads it like any query.
- **Dashboards are generated** (`python -m agent_platform.analytics dashboards`) and
  not versioned. D3's "Fleet page" is the **Agent Platform** dashboard. There is one
  dashboard per agent, and every dashboard is a Rill canvas with the same sections:
  outcomes, reliability, cost, tools, safety, grounding, plus the agent's own groups.
  Each measure is the catalogue's own `expr`, computed over the scope's raw rows, so it is
  exact for any range. Metrics carry a `label`, `hint` and `group`, and each view names its
  `count`.
- **Security and per-model metrics:** per-screen security metrics (PII, prompt injection,
  content moderation) and a `model_usage` view (every LLM and check model, with
  per-model latency and failures) feed the Security and Models sections. Models are
  plain views (materializing them only slowed start-up), and KPI tiles carry no sparklines
  (each section's trend charts do), to keep the pages light in the browser.
- **`hallucination_rate`** (unsupported ÷ checked claims) was added to the catalogue, next to
  `hallucination_turn_rate` (turns with at least one unsupported claim).
- **Legacy tables** are dropped by `make analytics-drop-legacy`, never by an export.
- **"Mart" is now "source".** An analytics source is a named dataset (`analytics.sources`):
  `live`, or `dummy`, the generated demo. Each is a DuckDB plus its Parquet export, and
  `analytics.source` picks the active one (`SOURCE=` on make). Identifiers were renamed
  to match (`ParquetExporter`, `SourceDatabase`, `parquet_dir`, `schema_version`,
  `source://manifest`), and the view field `source` became `reads`.
  `make analytics-mcp` was removed.
- **MCP transparency:** the platform's MCP client logs the handshake, the tool listing
  (descriptions and schemas), registration, and each call with its arguments, result and
  timing (`mcp.log`). `mcp/presentation.server_card` draws the MCP SERVER card.
- **Token-confidence step (logprobs):** `llm.logprobs` asks every completion for token
  log-probabilities. All three hosted NIM chat models in use return them (probed
  2026-10-09). `hallucination_protection/confidence.py` scores each reply sentence as the
  first step of the hallucination check, for CSA and RA (`ResponseGuardrail`) and for ops
  (before its figure check). A claim left undetermined in a low-confidence sentence counts
  as unsupported. It is level 1 of the hallucination check (`levels.py`), and travels
  first inside each agent's one `hallucination_check` event, before the agent's own
  level 2 (NLI entailment, the citation guard, or the ops figure check). Results: the
  level on that card and log line, a `logprob` check row, and the
  `low_confidence_rate` metric.
- **Ops cards follow CSA's.** The ops agent emits `llm_request` / `llm_response_received`,
  then `tool_request` · `gatekeeper_evaluation` · `tool_result`, then `hallucination_check`.
  Its own MODEL STEP, PROMPT, CONFIDENCE and FIGURES CHECK cards are gone; MCP SERVER stays.
