# CLAUDE.md — agentic-harness-core

Project instructions for Claude Code working in this repo. This is the production
codebase behind **The Agentic AI Systems Masterclass**, a 48+ episode YouTube
series aimed at CTO/CXO and Principal/Staff/Architect audiences. The codebase
itself is a core differentiator of the series — it must read as code written to
the standard of Anthropic, OpenAI, or Google, not tutorial-grade scaffolding.

## What this repo is

- A monolith containing:
  - `src/agent_platform/` — the shared technical platform library used from
    Episode 1 onward (logging, LLM adapters, persistence, memory, security,
    resilience, observability, audit, guardrails, hallucination protection,
    gatekeeper, tools, config, evals, and the terminal UI).
  - `src/customer_support_agent/` — the series' single recurring running
    project (a customer-support refund agent), which **composes**
    `agent_platform` rather than duplicating it. This grows incrementally
    across episodes through to the capstone.
  - `src/research_agent/` — the second agent on the same platform (scoped,
    durable web research), composed the same way.
  - Episodes are demonstrated by running these agents under config overlays
    (features, profiles, ablations), not by separate per-episode scripts.
    Each agent's `design_docs/patterns.md` maps the WF/AG patterns the series
    teaches to where that agent uses them.

## Non-negotiable engineering standards

- **Modularity**: small, composable, reusable pieces. Methods stay roughly
  5–20 lines. Every concern (logging, LLM interface, guardrails, memory,
  security, business logic, tool-calling, context management, DB interfaces,
  config) lives in its own module with a clear boundary.
- **Deep modules (Ousterhout)**: a lot of functionality behind a simple
  interface. Prefer fewer, well-hidden abstractions over many thin,
  leaky ones.
- **Abstraction via interfaces**: use ABCs/Protocols and dependency injection
  everywhere a component could reasonably be swapped (LLM provider, memory
  backend, DB). Don't hardcode concrete implementations into callers.
- **SOLID / DRY / YAGNI**, applied honestly — don't build abstraction for
  hypothetical future needs that aren't in the current episode's scope.
- **Resilience patterns**: circuit breaker, retry, timeout, saga, idempotency,
  bulkhead, and pub-sub all need real implementations in
  `agent_platform/resilience/`, not just folders. Ground correctness in the
  same references the series teaches from (Nygard, *Release It!*).
- **LLD patterns**: strategy, adapter, factory, builder, observer, facade,
  decorator — apply where they genuinely fit, not for their own sake.
- **Tests**: unit + integration tests for every module, runnable in CI.
  Coverage gate is 80%+ — do not let new code land under that without
  explicit sign-off.

## Design process (follow this order, don't skip ahead)

1. **Ubiquitous language first.** Before designing a new module, establish
   shared vocabulary for the domain concepts involved (DDD-style). Ask
   clarifying questions rather than assuming terms.
2. **Deep-module design** for the resulting boundaries.
3. **TDD as the coding discipline.** Resolve test-design decisions explicitly
   before writing implementation: unit granularity, what gets mocked, which
   behaviors are actually under test. Don't silently default these.

When planning a new module or feature, **interview-style, branch-by-branch
before presenting a finished plan** is the preferred mode — resolve one
design decision at a time rather than dumping a complete proposal. Ask before
building anything non-trivial.

## Stack

- **LLM providers**: NVIDIA NIM (OpenAI-compatible endpoint) as the cloud
  provider. Both go through `agent_platform/llm/` adapters — no
  direct SDK calls from business logic.
- **Observability**: LangSmith for LLM call tracing, plus the repo's own
  `observability/` + `audit/` modules for structured logs and the audit
  trail.
- **Orchestration**:LangGraph; Prefer LangGraph's built-in state management over hand-rolled state machines.
- **Terminal UI**: Rich + Textual, one unified app
  (`agent_platform/tui/app.py`) with panels/tabs for live trace, DB browser,
  memory inspector, and audit view — in the spirit of Claude Code / opencode.
  The TUI is a *consumer* of `observability`/`persistence`/`memory`/`audit`,
  never a second source of truth. File logging is required independently of
  whatever's on screen — the TUI does not replace it.
- **Config**: pydantic-settings with a `SECTION__FIELD` env var convention,
  except bare provider API keys (`nvidia_api_key`, `hf_api_key`), which stay
  as plain lowercase env vars by convention.
- **Memory**: all four CoALA classes are first-class and real, never
  stubbed — `working/`, `episodic/`, `semantic/`, `procedural/` under
  `agent_platform/memory/`.
- **Feature toggles**: each agent declares its capabilities in a
  `features:` block in its *own* `config.yaml`; `agent_platform` ships
  defaults and the `FeatureSet` mechanism (`config/features.py`) and
  never learns which agent is asking. Nine are toggleable —
  `basic_sanity`, `prompt_injection`, `content_moderation`, `pii`,
  `intent_classification`, `hallucination_guard`, `gatekeeper`,
  `memory` (+ its four CoALA children), `context` (+ `rerank`,
  `failure_detection`). Sub-features are parent-gated.

  **`basic_sanity` is the seam for an agent's own business logic.** The
  platform owns only `BasicSanityChecker` — a request in, findings out —
  and `BasicSanityRequest` carries `subject_id`, an opaque string, rather
  than a `customer_id`, so no agent is obliged to have customers or to
  number them. Customer Support supplies `OrderSanityCheck`; a new agent
  declares the feature in its own config and injects its own checker
  through `build_agent(basic_sanity=...)`, changing nothing here.

  **The harness is the set of declared features**: the controls you can
  remove and still get an agent that runs, just a worse or more
  dangerous one. Everything else — the control loop and its budgets,
  resilience, observability, audit, logging, persistence, tool calling,
  schema validation — is always on, because removing it does not produce
  a reckless agent, it produces a broken program.

  Read the tree **once, in the composition root**, and never inside a
  component. A component that consults configuration decides for every
  agent in the process, which is what made two postures in one process
  impossible. A withheld feature is a stage that is *never built* and so
  never appears in the compiled graph — not a component that runs and
  reports having done nothing.

  `/harness` is the only command that moves them: no args reports the
  tree, `on`/`off` moves all nine, `<path> on|off` moves one. It is a
  hot reload — nothing is read from disk, nothing is remembered, and a
  restart returns to what the config declared. `/config` stays read-only.

  **PRD G1 is retired as written.** The Gatekeeper may be withheld, and
  then a tool call executes with no authorization, budget or surface
  policy — but still through the Executor (idempotency, deadline). The
  audit entry is still written on that path, with standing UNGOVERNED, and
  no budget is committed.

  **The Gatekeeper decides authority only.** A fact about the *record*
  (the order exists, is the customer's, is refundable, the amount is its
  full total) belongs to the tool, checked once, with a conditional write.
  A fact about the *actor or the instruction* (offered, RBAC, mandate, the
  per-action limit) belongs to the Gatekeeper, which reads no database per
  call. Every decision names a standing: VOID (denied, nothing queued),
  RATIFIABLE (escalated to a person; approval is RATIFIED), VOIDABLE
  (allowed, and undoable until it settles — a refund is `issued`, then
  settles; `/void`, `/settle`), VALID. Never log or show a control state no
  code path produces or that cannot change a verdict.

## Known failure patterns — don't repeat these

These were found and fixed in past remediation passes. Watch for regressions:

- Episode code silently duplicating `agent_platform` (own LLM client, own
  auth checks, print-based logging) instead of composing it.
- Orphaned/dead files left behind after refactors (check for unreferenced
  modules before merging).
- `MemoryStore` (or any) Protocol declared but not actually implemented by
  its concrete classes — verify with mypy --strict, don't assume.
- CWD-relative file paths (e.g. policy files) that break outside repo-root
  invocation — use paths resolved relative to the package, not the shell's
  working directory.
- Config fields declared with no corresponding adapter built (e.g. local
  model settings with no `LocalLLMClient`) — don't add config surface ahead
  of the implementation.
- Runtime artifacts (`.coverage`, `.pytest_cache`, `data/*.db`, `logs/`)
  committed to git — these must be gitignored, not versioned.

## Verification standard

Don't report work as done from reading code alone — run it. `ruff`, `mypy
--strict`, and `pytest` (with coverage) are the bar for "verified," not
"looks correct."

## What NOT to do

- Do NOT hard code any value in the code, no hard coded strings or magic numbers.
- All values are strictly read from configuration files.
- Don't build ahead of the confirmed platform-first phasing: the full shared
  library ships before any episode-specific business logic beyond what's
  needed to exercise it.