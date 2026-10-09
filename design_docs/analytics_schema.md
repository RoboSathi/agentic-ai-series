# Analytics schema: facts, metrics, contracts and modules

The specification behind ADR 0009, in the vocabulary of
`analytics_ubiquitous_language.md`. It covers what each table holds, column by column;
which choke point writes it; how each Tier-1 metric is computed; how an agent
adds its own tables; what an eval reports; and the modules that implement this.

## 1. Conventions

- **Store:** the shared observability DuckDB (one writer: the agent's process).
  Readers use the Parquet export only.
- **Append-only.** No fact row is updated. Schema evolution is additive
  (`ADD COLUMN IF NOT EXISTS`); a column is never repurposed.
- **Common keys** (D5), first on every fact, stamped by the fact store from the
  bound context (never passed by hand):

| Column | Type | Meaning | NULL when |
|---|---|---|---|
| `agent_id` | VARCHAR | The agent (`agent.id`) | never |
| `session_id` | VARCHAR | The conversation | never inside a turn; `''` for work outside any session |
| `task_id` | VARCHAR | The task (explicit, or the turn's implicit task) | never inside a turn |
| `turn_id` | VARCHAR | The turn's trace id | `''` outside a turn (e.g. a background job) |
| `step` | INTEGER | The 1-based step within the turn | NULL outside a model-driven step (task events, screens before the first step) |
| `channel` | VARCHAR | `tui`, `eval`, `script`, … | never |
| `recorded_at` | TIMESTAMPTZ | When the row was appended (UTC) | never |

- **NULL vs 0.** NULL means not observed (the feature is withheld, or the agent
  has no such stage). 0 means observed and none happened. Every Tier-1 formula
  ignores NULLs.
- Table prefix `fact_`. Platform-owned tables never name an agent; agent-owned
  tables are prefixed with the agent's id (D7).

## 2. Facts (Tier 1)

### `fact_task_event` — grain: one task event
Written by the **task API** (`open_task`, `close_task`) and by the turn bracket for implicit tasks.

| Column | Type | Meaning |
|---|---|---|
| *common keys* | | `turn_id` = the turn in which the event happened |
| `event` | VARCHAR | `opened` or `closed` |
| `kind` | VARCHAR | The agent's name for the kind of task (`refund`, `research_run`, `direct_answer`, `implicit`); declared in config |
| `outcome` | VARCHAR | On `closed`: the agent's declared outcome; on `opened`: NULL |
| `success` | BOOLEAN | On `closed`: the outcome's declared flag |
| `terminal` | BOOLEAN | On `closed`: the outcome's declared flag |

The source view `tasks` joins opened and closed into one row per task: duration,
turns, steps, tokens and model calls rolled up from the facts below by `task_id`.
`status` is one of `closed`, `open` or `abandoned` (open longer than
`analytics.task_abandon_after`).

### `fact_turn` — grain: one turn
Written by the **turn bracket** when the turn ends, whatever ended it.

| Column | Type | Meaning | NULL when |
|---|---|---|---|
| *common keys* | | `step` NULL | |
| `status` | VARCHAR | `completed`, `crashed`, `cancelled` (platform-defined) | never |
| `latency_ms` | DOUBLE | Wall-clock duration of the turn | never |
| `steps` | INTEGER | Steps taken (logical model calls) | never (0 if no model call) |
| `step_cap` | INTEGER | The loop's step budget | the agent's loop declares none |
| `step_cap_hit` | BOOLEAN | Ended because the cap was reached (loop abort) | as above |
| `claims_checked` | INTEGER | Claims the grounding check judged | grounding withheld |
| `claims_supported` | INTEGER | Judged supported | grounding withheld |
| `claims_unsupported` | INTEGER | Judged unsupported (kept-and-flagged, repaired or removed) | grounding withheld |
| `screens` | VARCHAR[] | The screens the agent's composition root declares (`units.declare_screens`) | the agent declares none, or the turn carries no message (a resumed run) |
| `error` | VARCHAR | The exception type and message when `crashed` | otherwise `''` |

Totals per turn (tokens, calls, proposals, checks) are *not* stored here. The source rolls them up from their own facts by `turn_id`, so there is one source per number.

### `fact_model_call` — grain: one attempt
Written by the **LLM adapter** (`RecordedClient` inside each model's resilience stack).

| Column | Type | Meaning |
|---|---|---|
| *common keys* | | `step` = the step this attempt serves |
| `role` | VARCHAR | The role the model plays (`answerer`, `planner`, …; the provider name for the default client) |
| `provider` | VARCHAR | The configured provider |
| `model` | VARCHAR | The model that served the attempt |
| `attempt` | INTEGER | 1 for the first try of the step's call on any model; 2+ for retries and backups |
| `is_backup` | BOOLEAN | Served by a model after the role's primary |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | BIGINT | Usage billed |
| `latency_ms` | DOUBLE | The provider call's duration |
| `finish_reason` | VARCHAR | `stop`, `length`, `tool_calls`, …; `''` on failure |
| `succeeded` | BOOLEAN | False when the attempt raised |
| `error` | VARCHAR | The error when it failed |

### `fact_tool_call` — grain: one tool proposal
Written by the **single tool path** (§6.2) for every tool call a model proposes.

| Column | Type | Meaning | NULL when |
|---|---|---|---|
| *common keys* | | `step` = the step that proposed it | |
| `tool` | VARCHAR | The qualified name as proposed (`namespace.tool`, or the raw name if unknown) | never |
| `stage` | VARCHAR | `invalid`, `unknown`, `not_offered`, `refused`, `duplicate`, `executed` | never |
| `succeeded` | BOOLEAN | Execution succeeded | stage ≠ `executed` |
| `timed_out` | BOOLEAN | Execution hit its deadline | stage ≠ `executed` |
| `duration_ms` | DOUBLE | Execution time | stage ≠ `executed` |
| `side_effecting` | BOOLEAN | The tool writes | stage ∈ {`invalid`, `unknown`} |
| `reason` | VARCHAR | Why it stopped where it did (validation error, refusal reason, failure message) | never (`''` if executed and succeeded) |

### `fact_check` — grain: one check invocation
Written by the **check recorder** at three places: `MessageScreener`, the grounding protocol, and the model-dependency factories.

| Column | Type | Meaning | NULL when |
|---|---|---|---|
| *common keys* | | | |
| `check_name` | VARCHAR | `pii`, `injection`, `jailbreak`, `moderation`, `content_safety`, `grounding`, `logprob`, `nli`, `embedding`, `rerank`, `intent_model` | never |
| `purpose` | VARCHAR | `screen` (on the person's message), `grounding`, `dependency` | never |
| `backend` | VARCHAR | `rule`, `local_model`, `hosted_model`, `llm` | never |
| `model` | VARCHAR | The model that answered (`''` for a rule) | never |
| `verdict` | VARCHAR | `pass`, `flag`, `block`, `skipped`, `error`, `fail_open` | never |
| `score` | DOUBLE | The deciding score | the backend gives none |
| `latency_ms` | DOUBLE | Duration of the invocation | never |
| `items` | INTEGER | What it looked at (claims, entities, texts) | not countable |
| `flagged` | INTEGER | How many of those it flagged | not countable |

### Not Tier 1, kept
- `platform_audit_entries` (decisions). It stays as the audit trail and feeds agent pages, not the Fleet.
- `eval_score` (§5).

### Retired (dropped once the above exist)
`platform_llm_calls`, `platform_tool_calls`, `platform_turn_outcomes`, `platform_trace_spans` (as an analytics source), and the `turns`, `events`, `slo_*` views of ADR 0007.

## 3. Tier-1 metrics: the catalogue (`dim_metric`)
Each metric is defined once, in the platform semantic layer, by name, SQL,
direction and default SLO target. The Fleet page, the MCP `describe` output and
the ops agent all read this catalogue. Runtime metrics filter
`channel <> 'eval'` by default. In the formulas below, `t` is the source's `turns`
view, `k` is `tasks`, `m` is `model_calls`, `p` is `tool_calls` and `c` is `checks`.

| Metric | SQL (per `agent_id`, per period) | Better | NULL when |
|---|---|---|---|
| `resolution_rate` | `avg(success::DOUBLE) FILTER (WHERE status='closed' AND terminal)` over `k` | higher | no terminal task |
| `availability` | `1 - avg((status='crashed')::DOUBLE)` over `t` | higher | — |
| `turn_latency_p50/p95/p99_ms` | `quantile_cont(latency_ms, q)` over `t` | lower | — |
| `task_latency_p50/p95/p99_ms` | `quantile_cont(duration_ms, q) FILTER (WHERE status='closed')` over `k` | lower | — |
| `model_latency_p50/p95/p99_ms` | `quantile_cont(latency_ms, q)` over `m` (by `model`, `role`) | lower | — |
| `tokens_per_task` | `sum(total_tokens) / count(DISTINCT task_id)` over `m` | lower | — |
| `tokens_per_resolved_task` | tokens of resolved tasks ÷ resolved tasks (`k` ⨝ `m`) | lower | no resolved task |
| `tokens_per_turn` | `sum(total_tokens) / count(DISTINCT turn_id)` | lower | — |
| `steps_per_turn_p50/p95` | `quantile_cont(steps, q)` over `t` | lower | — |
| `steps_per_task_p50/p95` | over `k` | lower | — |
| `loop_abort_rate` | `avg(step_cap_hit::DOUBLE)` over `t` | lower | no step cap declared |
| `retry_rate` | `count(*) FILTER (WHERE attempt > 1) / count(DISTINCT (turn_id, step, role))` over `m` | lower | — |
| `fallback_rate` | `avg(is_backup::DOUBLE)` over `m` | lower | — |
| `model_calls_per_task` | `count(*) / count(DISTINCT task_id)` over `m` | lower | — |
| `tool_call_validity` | `avg((stage='executed')::DOUBLE)` over `p` (+ breakdown by `stage`) | higher | no proposals |
| `tool_error_rate` | `avg((NOT succeeded)::DOUBLE) FILTER (WHERE stage='executed')` over `p` (by `tool`) | lower | nothing executed |
| `duplicate_call_rate` | `avg((stage='duplicate')::DOUBLE)` over `p` | lower | no proposals |
| `dependency_error_rate` | `avg((verdict IN ('error','fail_open'))::DOUBLE) FILTER (WHERE purpose='dependency')` over `c` (by `check_name`, `model`) | lower | no dependency calls |
| `dependency_latency_p95_ms` | same filter, `quantile_cont(latency_ms, 0.95)` | lower | — |
| `screen_coverage` | turns with ≥1 `purpose='screen'` row for check X ÷ turns, for the checks the agent declares | higher | screen withheld |
| `block_rate` | `avg((verdict='block')::DOUBLE) FILTER (WHERE purpose='screen')` (by `check_name`, `backend`) | — (watched) | screen withheld |
| `flag_rate` | `avg((verdict='flag')::DOUBLE)`, same filter | — (watched) | screen withheld |
| `guard_failure_rate` | `avg((verdict IN ('error','fail_open'))::DOUBLE) FILTER (WHERE purpose='screen')` | lower | screen withheld |
| `groundedness` | `sum(claims_supported) / sum(claims_checked)` over `t` | higher | grounding withheld |
| `pii_detection_rate`, `injection_block_rate`, `injection_guard_failure_rate`, `moderation_block_rate`, `moderation_flag_rate`, `moderation_guard_failure_rate` | per screen family, per message (distinct turns) over `c` | watched / lower | that screen withheld |
| `pii_coverage`, `injection_coverage`, `moderation_coverage` | `avg(ran)` per screen family over `screen_coverage` | higher | that screen withheld |
| `call_latency_p50/p95/p99_ms`, `call_failure_rate`, `tokens_per_call` | over `model_usage` (every LLM attempt + every model a check consulted), by `model`, `kind` | lower | — |
| `hallucination_rate` | `sum(claims_unsupported) / sum(claims_checked)` over `t` | lower | grounding withheld |
| `hallucination_turn_rate` | `avg((claims_unsupported > 0)::DOUBLE) FILTER (WHERE claims_checked > 0)` over `t` | lower | grounding withheld |
| `low_confidence_rate` | `flagged / items` of `logprob` checks over `c`: sentences the model wrote with low token confidence | lower | logprobs not returned |

SLO targets per metric live in `observability.slo_targets`, keyed by these names.
Error budgets are computed over `analytics.slo_windows_days` exactly as today.

## 4. Tier 2: agent-owned tables (D7)
- An agent defines its tables in its own package (e.g.
  `customer_support_agent/analytics/`). Each table is prefixed with the agent id and
  carries the common keys, so it joins the Tier-1 facts by `task_id` or `turn_id`.
- The agent writes them through the platform **fact store** (§6.4), so stamping,
  additive evolution and never-raise behaviour are shared. The table definitions
  and the decision about what to record are the agent's.
- The agent ships its semantic layer at `src/<pkg>/analytics/semantic_layer.yaml`, found
  by convention (`analytics.agent_layers_glob`) so the source holds every agent whichever
  process exports it. The file has the same shape as the platform's (views, SQL, column
  docs, metrics) plus `agent_id`; its metrics become `dim_metric` rows with `tier: 2` and
  `owner: <agent_id>`. A view whose tables do not exist yet is skipped with its reason.
- The exporter, the MCP server (views namespaced `<agent_id>__<view>`) and the
  dashboard generator read registered layers next to the platform's. The
  boundary test still forbids agent names in platform files.

## 5. Tier 3: the eval contract (D8, shape only; suites come in the evals rework)
`eval_score`, grain: one eval run × case × metric.

| Column | Type | Meaning |
|---|---|---|
| `agent_id`, `channel` (= `eval`) | VARCHAR | As runtime |
| `run_id`, `suite`, `case_id` | VARCHAR | Which run, which golden set, which case |
| `task_id`, `turn_id` | VARCHAR | The runtime task and turn the case produced (eval traffic writes Tier-1 facts too), so cost and latency per case come from the same facts |
| `metric` | VARCHAR | A core metric (`task_success`, `groundedness`, `tool_selection_accuracy`, `attack_success_rate`) or the agent's own |
| `core` | BOOLEAN | One of the four core metrics |
| `score` | DOUBLE | 0–1 |
| `passed` | BOOLEAN | Against the metric's threshold |
| `severity` | VARCHAR | `hard` (must pass) / `soft` |
| `grader` | VARCHAR | `oracle`, `judge:<model>`, `rule` |
| `recorded_at` | TIMESTAMPTZ | |

Every agent's eval page shows the four core metrics in the same place. Its own
metrics appear below them.

### 3.1 How a metric is defined (as built)
Each catalogue entry is `{name, label, hint, group, view, expr, better, description, null_when, by}`
(`label (hint)` is how a dashboard names it, and `group` is its dashboard section): one
aggregate over one view, reported per `agent_id` and per breakdown in `by`. It is
rendered three ways from that one definition: per day (`metric_daily`), per SLO
window (`metric_window` → `slo_error_budget`), and per period on demand (MCP
`compute_metric`, `/stats`). Two helper shapes keep every metric in this one form:
`tasks` carries each task's tokens, and the `screen_coverage` view has one row per
turn per declared screen. The Fleet dashboard is generated from the same entries.

## 6. Modules (deep, small interfaces)

### 6.1 Turn and task bracket (`agent_platform/observability/units.py`)
- **Interface:** `turn_trace(...)` (exists; extended) and, inside it,
  `open_task(kind) -> task_id`, `close_task(outcome)`, `current_units()`.
- **Hides:** binding of the turn/task/step context variables; turn timing; crash
  and cancel detection (exception and cancellation in the bracket); implicit
  tasks; the outcome declaration lookup and flag stamping (an undeclared outcome
  raises at start-up validation, not at runtime); writing `fact_turn` and
  `fact_task_event`; abandoning tasks at session end.
- The step counter lives here too: the outermost model client calls
  `next_step()`, and the loop guards report `step_cap_hit`.

### 6.2 Single tool path (`agent_platform/tools/path.py`)
- **Interface:** `ToolPath.run(proposal: ModelToolCall, turn: ToolTurn) -> ToolOutcome`.
  `ToolTurn` carries what the caller owns: the turn's control loop and step budget, the
  authority builder (which names what was offered), an optional argument preparation,
  card hooks, and whether a failed read may be retried. "Offered" is ruled on by the
  Gatekeeper (its `offered` question → stage `not_offered`); a withheld Gatekeeper
  checks nothing (CLAUDE.md).
- **Hides:** registry lookup (unknown vs invalid, with the validator's reason);
  offered-set check; duplicate suppression (`ToolControlLoop`); the optional
  Gatekeeper and audit (including UNGOVERNED); the Executor (deadline,
  idempotency); writing `fact_tool_call` at whichever stage the proposal stopped.
- CSA `act.py`, RA `answer_nodes._execute` and Ops `GovernedCalls` collapse into
  calls to it. Their cards stay theirs.

### 6.3 Check recorder (`agent_platform/observability/checks.py`)
- **Interface:** `record_check(check, purpose, backend, model, verdict, latency_ms, score=None, items=None, flagged=None)`,
  `record_screen`, `record_pii`, `record_grounding`, and a decorator
  `recorded_dependency(check, method, backend)` for the four model factories.
- **Hides:** mapping `ScreenOutcome.checks` dicts and `ResponseGuardrailResult`
  onto rows; timing; fail-open detection.

### 6.4 Fact store (`agent_platform/observability/facts.py`, reworked)
- **Interface:** `FactStore.append(fact)` (and `open_task(agent, session)`, the one read
  the bracket needs), plus typed fact models. The composition root installs the process's
  store (`install_fact_store`); choke points append to it.
- **Hides:** stamping the common keys from context; the table DDL derived from
  the model; additive evolution; never-raise.
- It is used by §6.1–6.3 and by agents for their Tier-2 tables.

## 7. Test design (decided before code)
- **Unit tests, per module.** The bracket, the tool path, the check recorder and the
  fact store each get tests against a real in-memory DuckDB. Nothing in the store
  is mocked.
- **Mocked: the LLM only** (scripted models). Hosted guards and model
  dependencies are replaced by fakes that implement their protocols, so verdict,
  error and fail-open paths are exercised deterministically.
- **Under test:**
  - every stage a proposal can stop at;
  - every turn status, including crash and cancel;
  - implicit vs explicit tasks, and abandonment;
  - an undeclared outcome failing validation;
  - NULL vs 0 for withheld features;
  - each Tier-1 formula against hand-computed values on a fixture fleet of two fake agents (one with every feature, one with none), which also proves the "any agent" claim.
- **Contract tests:**
  - every `dim_metric` row's SQL runs on the fixture source;
  - every documented column exists;
  - the Rill Fleet page names only catalogue metrics;
  - no platform file names an agent.
- **Integration:** one scripted end-to-end turn per real agent (CSA, RA, Ops)
  asserts that its facts appear with the right keys.
