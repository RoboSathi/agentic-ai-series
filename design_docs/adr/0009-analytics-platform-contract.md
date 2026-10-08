# ADR 0009 — Analytics is a platform contract

- Status: proposed (design review before any code)
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
