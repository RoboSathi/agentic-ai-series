# Research Agent (`ra`) — Build Plan

Phase 0 output. Written after reading the platform code named in the
build spec §0.2, before any feature code. It records what already exists
(and is reused), what is extended, what is new, where the spec and the
repository disagree, and the order of work.

## 1. Reuse map

Legend: **reuse** = used as-is · **extend** = generic change to an existing
commons module · **new** = new domain-free commons mechanism · **agent** =
lives in `research_agent` (pattern composition, research vocabulary).

| Mechanism (spec §3/§4) | Found in the repo | Decision |
|---|---|---|
| Model routing per role | `llm/client.py` has `create_pinned_llm_client(provider, model)`; `llm/cascade.py` is *escalation* (cheap tier first), not routing | **new** `llm/roles.py` `RoleModelRouter` built on `create_pinned_llm_client`; config `llm.roles` (empty in the platform, filled by the agent). Cascade is not a fit: it answers "which tier is good enough", not "which model plays this role". |
| Budgets | `control_loop/watchdogs.py` = duplicate-action + iteration ceiling; `WatchdogLayer` names `token_budget`/`cost_budget` but nothing implements them; `gatekeeper/ledger.py` = money window per action class | **new** `control_loop/budget.py` `BudgetGuard` (reserve → commit/release, hard caps, exhaustion event) reporting through the existing `ControlCheck`/`ControlLoopReport` types, so a budget stop reads like any other control-loop stop. Gatekeeper ledger is a cumulative *money* ledger across sessions — not reused. |
| Durable graphs | `durability/` `DurableGraphRunner`, `RunRegistry`, `create_checkpointer`, `model_types` | **reuse** for both ra graphs. |
| HITL interrupt | csa `graph/nodes/review_wait.py::await_approval` — its generic core is `interrupt({WAITING_ON, SUMMARY, …})` | **extend** `durability/hitl.py` `await_decision(waiting_on, summary, **fields)`. ra uses it for every interview/rubric turn (the scoping graph *pauses on the user*). csa is **not** migrated to it (would touch csa). |
| Isolated sub-graph | none | **new** `durability/isolated.py` `IsolatedGraphRunner` (fresh thread, in-memory checkpointer, explicit inputs only). The Independent Audit runs on it. |
| Prompt-injection classifier | `security/prompt_injection/detector.py` `InstructionDataSeparator.find_phrases` / `separate` | **reuse** as layer 2 (classifier) over fetched text and for delimiting untrusted text. Agent-specific phrases are added in ra's overlay (`security.prompt_injection.suspicious_phrases`). |
| Authority ranking | `gatekeeper.authority_hierarchy` already ranks `untrusted_content` last | **reuse** the configured hierarchy; ra tags every fetched span with the bottom rank. |
| NLI | `evals/entailment.py` `EntailmentModel`, `create_entailment_model()` (local DeBERTa → hosted → lexical); `hallucination_protection.DetectionLadder` wraps it with retrieval over chunks | **reuse** `EntailmentModel` for claim→span support (the span is already *the* evidence, so the ladder's retrieval stage adds nothing). |
| Structured output | `guardrails/guards.py` `PydanticValidator.validate_or_raise` | **reuse** for every ra LLM role (one `StructuredCall` helper in ra). |
| Canonical hash | `evals/contracts.py::canonical_hash` | **extend**: moved to `agent_platform/primitives/hashing.py` (neutral layer) and re-exported from `evals.contracts` so nothing else changes. The artifact store must not depend on the eval package. |
| Resilience | `resilience/` `ResilientCaller` (bulkhead → breaker → retry → timeout), `CircuitBreakerImpl`, `rate_limiter` (host-bucketed httpx transport via `llm/http.py::create_http_client`) | **reuse**. Web calls go through one `ResilientCaller` per provider and the rate-limited httpx client. **extend** platform config: `tavily` bucket in `resilience.rate_limit.buckets`. |
| Idempotent fetch | `resilience/idempotency` (SQLite, per-agent) | **reuse the idea, not the table**: the fetch cache is URL-keyed inside `WebGateway`; a replayed fetch never repeats. |
| Web access | none | **new** `web/` package: `WebGateway` (`search()`, `fetch()`, modes live/record/replay), `SearchProvider` + `TavilyProvider`, `Fetcher` + reader fallback, `FixtureStore`. |
| Evidence ledger + verbatim rule | none | **new** `evidence/` package: `EvidenceLedger` (append-only), `VerbatimSpanValidator`. Generic claim/source/span vocabulary. |
| Frozen artifacts | none (evals hash cases, nothing freezes/writes typed artifacts) | **new** `artifacts/` package: `FrozenArtifact`, `DirectoryArtifactStore` (run dirs, verify-unchanged). |
| Async DAG scheduler | none (`design_patterns/workflow/fan_out_fan_in.py` is a teaching example, not a platform scheduler) | **new** `scheduling/` package: `TaskGraph` (cycles, waves, blocked dependents) + `AsyncDagScheduler` (bounded semaphore, failure isolation). LangGraph mapping: one `Send` per ready node per wave; the scheduler's semaphore bounds the Sends. |
| Rubric-check registry | `verification/` is read-back of side effects; `evals/` grades transcripts offline | **new** `verification/acceptance.py` `AcceptanceCheckRegistry` (named checks by kind `deterministic|nli|judge`, evaluate a list of check specs against a context). Lives beside `verification/` because it verifies an artifact at run time, not a transcript offline. |
| Memory | `memory/` four stores + `MemoryFacade` (working/episodic/semantic/procedural; absent store = not built) | **reuse**: working = live run state (`note`), episodic = run index (`record_event`), semantic = archived artifacts (`submit` create), procedural = **not built** (`memory.procedural: false`). |
| Audit | `audit/` `AuditLedger` (`AuditEntry` allow/deny/escalate) | **reuse**; every transaction (contract frozen, search, fetch, gate decisions, security rejections, final status, toggle snapshot) is one entry. |
| Stage feed | `observability.StageReporter` | **reuse**; ra stage keys labelled via `AgentHost.stage_labels()`. |
| Shell | `host/protocols.py` `AgentHost`, `tui/app.py` | **extend** with optional hooks (`greeting`, `side_panels`, `cancel`) and `ConsoleEffect.agent_turn` (a slash command that is a conversational act is forwarded to `handle()`). csa implements none of them and is unaffected. |
| Eval pipeline | `evals/` plugin seam, stages, store, reporter, `RubricJudge` | **reuse**; **extend** two optional plugin hooks: `native_security_suite()` (run the security suite as native golden cases instead of Promptfoo) and `judge_client()` (the plugin may supply the judge's client — ra supplies a deterministic one in fixture mode). |
| Feature toggles | `config/features.py` `FeatureSet` — a generic tree; agents declare their own keys | **reuse** unchanged: ra declares its eight keys in its own overlay. |
| Logging colours | `logging/core.py` `SUBSYSTEM_COLORS` | **extend** with `scheduler, budget, web, ledger, artifacts, research_agent` (an agent logger name, registered the same way `customer_support_agent` already is). |

## 2. Conflicts between the spec and the repository (and how each is resolved)

1. **Branch name.** The spec asks for `feat/research-agent`; this session is
   pinned to `claude/research-agent-monolith-vy9ct7`. All work is on the
   latter.
2. **Baseline `make verify` is red on `main`.** 56 files fail
   `ruff format --check`, `mypy --strict` reports 44 errors in 25 files
   (commons, csa, episodes, design_patterns), and two test modules fail to
   import (`is_customer_ready`, and `test_design_gaps.py`), which aborts
   collection so no test runs. "Never move on while verify is red" cannot
   be honoured literally from phase 1. Resolution: a **baseline-repair
   commit before phase 2**, strictly behaviour-preserving (formatting,
   type annotations/casts, stale test imports). Where it touches csa it is
   typing only; every such edit is listed in ADR 0005.
3. **`tests/` already exists** (the spec expected none). Conventions
   followed: `tests/unit/…`, `tests/integration/…`, `ScriptedLLM` in
   `tests/conftest.py`. New tests go in `tests/agent_platform/…` and
   `tests/research_agent/…` as the spec asks; both are ordinary packages
   under the same conftest.
4. **Banned words already in commons.** `rubric` appears in the eval
   judge (`evals/contracts.py::Rubric`, `graders/rubric_judge.py`) and
   `report` is everywhere (`StageReporter`, `evals/reporting`). A grep for
   either across all of commons would fail on code that predates ra.
   Resolution: the guard test bans `archetype`, `research`,
   `brief` everywhere in commons (one pre-existing English use of "brief"
   is allow-listed by file+line), and bans `rubric`/`report` inside every
   commons module *added for ra*. Recorded in ADR 0003.
5. **PRD/HLD/LLD/Program_Design are in this repo** (`design_docs/`), not
   absent as the spec assumed. The top-level `design_docs/README.md`
   indexes them and notes they predate ra and must be reconciled by a
   human; they are not edited.
6. **Promptfoo security stage needs npx + network**; ra's security suite is
   seeded hostile pages that must run offline. Resolution: optional plugin
   hook `native_security_suite()` (see §1). csa still uses Promptfoo.
7. **No model credentials in this environment** (`nvidia_api_key` unset)
   and the spec requires `make agent ra` to run the whole flow in replay and
   evals to be deterministic. Resolution: `research_agent.model_backend:
   fixture | routed`. `fixture` is a deterministic reference model over
   the synthetic fixture world (it reads the same prompts a real model
   would and answers each role's schema); `routed` uses `llm.roles`
   through `RoleModelRouter`. Demo/eval posture defaults to `fixture` +
   `web.mode: replay`; ADR `ra-0015`.
8. **Tavily docs host is blocked** by this sandbox's egress proxy.
   The request/response shape was verified from the official docs'
   published reference (docs.tavily.com, via search index): `POST
   https://api.tavily.com/search`, `Authorization: Bearer <key>`, body
   `query, search_depth, max_results, include_answer, include_raw_content,
   include_domains, exclude_domains, time_range, topic`, response
   `{query, answer, results[{title,url,content,score,raw_content,
   published_date}], response_time, request_id}`. The adapter tolerates
   missing optional fields. Recorded in ADR `ra-0006`.
9. **Platform features ra does not compose** (`basic_sanity`,
   `gatekeeper`, `intent_classification`, `hallucination_guard`,
   `context`) are inherited from the platform default tree by deep merge
   and cannot be deleted by an overlay. ra declares them `false`; they
   are listed by `/harness` and moving them changes nothing for ra.
   Recorded in `ra-0016`.
10. **Slash commands never reach the agent** in the current console, but
    `/done`, `/waive`, `/rubric`, `/approve` are conversational acts that
    change the interview. Resolution: `ConsoleEffect.agent_turn` — the
    command validates its syntax and forwards the line to `handle()`,
    which is also the path the eval `ScriptedUser` drives. ADR 0004.

## 3. Phases

| # | Phase | Exit criterion |
|---|---|---|
| 0 | Discovery, this plan | plan committed |
| 0.5 | Baseline repair (formatting, types, stale tests) | `make verify` green on unchanged behaviour |
| 1 | Makefile agent registry + stub `research_agent` (config, `ResearchHost`) | `make agent ra` boots; `make agent csa` / `make eval-oracle csa` unchanged |
| 2 | Commons mechanisms, one at a time with tests; then the ra walking skeleton (every node a stub that reports through `StageReporter`) | verify green |
| 3 | Scoping: gate, checklists, interviewer, hybrid termination, `/done`, `/waive`, challenge check, checklist panel | verify green |
| 4 | Rubric: templates, generator, editor, freeze + hash | verify green |
| 5 | Research core: planner + validation, scheduler wiring, worker, `WebGateway`, three defences, ledger + verbatim rule | verify green |
| 6 | Reflection, synthesis, Independent Audit, gap-fill, partial delivery, archive, cancel | verify green |
| 7 | Evals: fixture world, plugin, oracle, security suite, judge rubric, planted flaws, ablations | `make eval ra` oracle at 100% |
| 8 | Docs, ADRs, `patterns.yaml`, `make patterns`, final verify, optional `/recall` | DoD (spec §10) |

## 4. Key design choices made here (details in the ADRs)

- **Session phase lives in LangGraph state.** The scoping graph is one
  durable thread per scope that *interrupts on the user* (`await_decision`)
  each turn; `handle()` resumes it with the next message. No hand-rolled
  session machine.
- **Waves are LangGraph supersteps.** `plan_wave` computes the ready set
  from `TaskGraph`; a conditional edge fans out one `Send` per node; the
  worker holds a slot of the scheduler's semaphore; `reflect` runs once
  after the superstep. Checkpoints land at wave boundaries.
- **Cancel is cooperative.** `ResearchHost.cancel()` sets the run's
  cancellation event; workers check it before every search/fetch and the
  wave router checks it between waves. Status becomes `CANCELLED`, the
  checkpoint stays.
- **Budgets reserve before every await**: `BudgetGuard.reserve()` is
  synchronous check-and-decrement, so parallel workers on one event loop
  cannot overshoot.

## 5. As built: deviations and deferrals

Recorded at the end of the build; the ADRs hold the reasoning.

### Deviations from the build brief

1. **Branch**: `claude/research-agent-monolith-vy9ct7`, not `feat/research-agent` (§2.1).
2. **Baseline repair before any phase** (ADR 0005): formatting, typing,
   stale imports. On instruction, `episodes/` was excluded from lint, types
   and tests (they will be rewritten), and 152 stale test functions that
   exercised removed APIs were deleted. This touched csa beyond the
   registry/docs/patterns scope: typing-only edits, plus one crash fix in
   `graph/nodes/act.py` (an audit entry pydantic rejected on the
   unverified-side-effect path). Each is listed in ADR 0005.
3. **Commit granularity**: phases 0, baseline, 2 (registry + stub), 3
   (commons mechanisms), 4–7 (scoping, rubric, research core, reflection /
   synthesis / audit / archive / cancel), 8 (evals) and 9 (docs) are
   separate commits, each verified green in a clean worktree. Phases 4–7
   are **one** commit: they were built as one vertical slice against the
   walking skeleton, which was replaced in place rather than committed on
   its own.
4. **Vocabulary guard** is narrower than asked: `report` and `rubric`
   predate ra in commons (ADR 0003).
5. **Model backend** `fixture | routed` (ra-0015); **platform features
   declared false** (ra-0016).
6. **Security stage** is a native golden-style suite graded by the oracle,
   not Promptfoo; **DeepEval** runs no metrics (hosted models only); the
   **judge** stage uses an offline `ReferenceJudge` under the fixture
   backend, and the run record still names the platform's configured judge
   model (the platform records config, not the injected client).
7. **Eval store and reports** are ra's own (`data/research_evals/`,
   `reports/research_evals/`), because suite names are per agent.
8. **Evidence ledger identity** (commons): one span of one page is one
   entry regardless of the sub-question that found it.
9. **Defects the evals found and fixed in the agent** (phase 8): a scope
   closed before any criterion was delivered as an empty `COMPLETE`
   (fixed by a findings rule on every rubric); stale sources were read
   despite a recency window (now out of scope for the worker); an empty
   gap-fill plan was recorded as run; the explainer's user criteria mapped
   to no rubric item; numbers with thousands separators were split as
   separate answers; the ledger admitted the same span twice.

### Deferred

- **`/recall`** (optional in the brief): not built. The archive's episodic
  index and semantic fact are the substrate it would read.
- **Resuming an interrupted autonomous run after a process restart** from
  the console: checkpoints are written at every wave boundary and the run
  registry records the thread, but no console command resumes a run.
- **Live Tavily, DuckDuckGo and real-model runs** are the default since
  ADR ra-0018 but are not exercised in this environment (no credentials, no
  outbound search): the adapters, router, quota and cache are tested with
  mock transports and a scripted DuckDuckGo backend.
- **`fixed_loop`** is the one catalogue pattern neither agent uses
  (`design_docs/patterns_coverage.md`).
