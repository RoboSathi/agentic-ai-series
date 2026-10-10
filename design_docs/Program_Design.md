# Program Design — Agentic AI Systems Platform (`agentic-harness-core`)

**Companion documents:** `PRD.md`, `HLD.md`, `LLD.md`.
**Audience:** primarily a coding agent (e.g. Claude Code) executing the
build in phases; secondarily Karan and collaborators as reference.

---

## 1. Repo layout (target state — extends the confirmed monolith)

```
agentic-harness-core/
├── src/
│   ├── agent_platform/
│   │   ├── config/
│   │   ├── logging/
│   │   ├── llm/
│   │   ├── persistence/
│   │   ├── audit/
│   │   ├── observability/
│   │   ├── context/
│   │   ├── memory/
│   │   │   ├── working/  ├── episodic/  ├── semantic/  └── procedural/
│   │   ├── guardrails/
│   │   ├── hallucination_check/
│   │   ├── gatekeeper/
│   │   ├── security/
│   │   │   ├── auth/  ├── pii/  ├── prompt_injection/  └── content_moderation/
│   │   ├── resilience/
│   │   │   ├── circuit_breaker/  ├── retry/  ├── timeout/  ├── bulkhead/
│   │   │   ├── saga/  ├── idempotency/  └── pub_sub/
│   │   ├── verification/            # effect read-back — see LLD §24
│   │   ├── host/                    # AgentHost protocols + loader: how an agent reaches the shell
│   │   ├── primitives/              # money, hashing: layer-0 value types
│   │   ├── presentation/            # text_tree, text_table: plain-text renderers
│   │   ├── sandbox/                 # [new — parked, TODO]
│   │   ├── control_loop/            # [new — parked, TODO; merges the earlier loop_engineering + state_management split]
│   │   ├── mcp/                     # [new — parked, interface-only; see LLD §16.1]
│   │   ├── multi_agent/             # [new — stub only, design parked]
│   │   ├── prompt_compilation/      # [new — parked, TODO]
│   │   ├── evals/
│   │   ├── tools/
│   │   ├── tui/                     # live status feed only this phase — DB browser/memory inspector/audit view deferred
│   │   └── data/                    # seed/fixture data, not runtime state
│   └── customer_support_agent/
│       │   # Each folder pairs with the agent_platform package it
│       │   # composes, so the mapping is visible in the tree rather
│       │   # than only in the imports. `core/` is gone: it had become
│       │   # fifteen unrelated concerns and a 2,814-line graph.py.
│       ├── domain/        # Order, User, HitlRequest, identity
│       ├── persistence/   # schema.sql, repositories
│       ├── security/      # authentication
│       ├── gatekeeper/    # business_rules, mandates
│       ├── resilience/    # refund_saga
│       ├── verification/  # refund_effects — the read-back for this agent
│       ├── tools/         # definitions, handler
│       ├── intent/        # router
│       ├── deterministic_checks/  # order_validator
│       ├── context/       # context sources
│       ├── control_loop/  # quality_gate (was deliberation)
│       ├── observability/ # accounting
│       ├── presentation/  # orders, reporting
│       ├── console/       # this agent's slash commands + its AgentHost
│       ├── review/        # the human side of escalation
│       ├── graph/         # state, edges, builder, agent, nodes/
│       ├── prompts/
├── tests/
│   ├── unit/               # mirrors src/ package structure 1:1
│   └── integration/
├── logs/                   # gitignored
├── .github/workflows/      # CI — see §5
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── CLAUDE.md
├── README.md
├── PRD.md / HLD.md / LLD.md / Program_Design.md
```

**Rule:** `tests/unit/` mirrors `src/` 1:1 — every `src/agent_platform/x/y.py`
has a `tests/unit/test_x_y.py`. This makes coverage gaps and orphaned files
(both caught once already) trivially greppable.

## 2. Ubiquitous language glossary

Established jointly (Karan as domain expert, Claude as developer) before
any new module design, per this repo's own DDD-first process rule. This
glossary is the single source of truth for terms used across
`PRD.md`/`HLD.md`/`LLD.md` — code identifiers should match these terms
exactly (a class named something a viewer wouldn't recognize from the
series is a naming bug).

| Term | Definition |
|---|---|
| **Agent** | Model + Harness (E01). Never used to mean "the model" alone. |
| **Harness** | Everything surrounding the model that turns a proposal into a governed action. |
| **Authority** vs **Agency** | Agency is the delegation of reasoning; Authority is the delegation of the right to act. An agent claiming authority doesn't create it (E02). |
| **Authority Hierarchy** | provider > org > product > workspace > user_task > runtime_reminders > tool_observations > untrusted_content. Only ranks 1-5 grant/expand authority. |
| **Sandbox → Gatekeeper → Execution** | The three-stage control plane (E01, E12). |
| **Surface Class** | Locked / Editable / Append-only / Human-controlled — what kind of resource is being touched (E12). |
| **Tier-1 / Tier-2** | Tier-1 = deterministic, authoritative, can block. Tier-2 = semantic/LLM-based, advisory-only, never gets veto power ("the Semantic Trap," E12). |
| **Automation Level** | Sheridan's 1-10 scale, applied per action class (E03). |
| **Three Gates** | Path Cardinality × Blast Radius × Certifiability — the scoring framework that sets Automation Level (E03). |
| **Structuring attack** | Many individually-authorized actions summing to an unauthorized outcome (E02). |
| **Working / Episodic / Semantic / Procedural memory** | The four CoALA stores (E13) — Working = active context (a pointer, not a store), Episodic = what happened, Semantic = what's generally true, Procedural = how to do things. |
| **Context Assembly pipeline** | gather_candidates → score_relevance → fit_to_budget → format → order (E08). |
| **Context failure** | Rot, Distraction, Confusion, Clash, Poisoning — five distinct degradation modes (E17). |
| **NOLLM Scoring Cascade** | exact_assertion → deterministic_rule → embedding_similarity → nli_entailment → calibrated_llm_judge → human_adjudication (E10). One implementation, reused by evals/context/tools/hallucination_check/llm (model cascading). |
| **Eval Pyramid** | component → integration → E2E golden-task → production shadow (E09). |
| **LoopController** | Six-layer watchdog: step_count, wall_clock, token_budget, cost_budget, duplicate_action, semantic_stagnation (E20). |
| **FSM state table** | IDLE → PLANNING → ACTING → WAITING_TOOL → WAITING_HUMAN → DONE → FAILED (E23). |
| **Checkpoint** | `(state, transition_history, durable_facts)` — never an arbitrary snapshot (E23). |
| **Circuit Breaker / Bulkhead / Saga / Idempotency / Retry / Timeout / Pub-Sub** | The seven resilience patterns (E12 general treatment, E35 full build-out), grounded in Nygard's *Release It!*. |
| **PII Filter** | Detection / Redaction (mask, tokenize, remove) / Minimization, applied at input, memory, and log surfaces (E32). |
| **Trace** | One hop: trace_id, actor, authority_context, tool, policy_version, model_version, state_transition, result, latency, cost (E36). |
| **SLO / Error Budget** | Agent-specific SRE targets (E37): task success, unauthorized-action rate, hallucination rate, escalation rate, loop-abort rate, tool failure rate, cost/success, recovery success, policy-violation rate. |
| **RBAC / PoLP** | NIST role model + Principle of Least Privilege; just-in-time scoped elevation only, never standing (E10 in v6 numbering). |
| **Topology** | Supervisor / Hierarchy / Peer Network / Swarm — multi-agent coordination shape (E41). |
| **Blast Radius** | Scope of damage a wrong action can cause — one of the Three Gates variables, also drives `sandbox` isolation-tier selection (E03, E34). |
| **Tool-Use Loop** | The `reason ⇄ act` cycle: the model proposes a *typed tool call*, the harness executes it, the observation returns. Bounded by `max_tool_iterations`. Deliberately **not ReAct** — there is no natural-language thought step, and the proposal is a validated call rather than parsed prose. The research agent's `research/worker.py` is the ReAct form. |
| **Output Quality Gate** | Plan lookup → deterministic critique → at most one repair → independent score. A single pass, not a loop; named `quality_gate` after being called a "deliberation loop" it never was. |
| **Action Mandate** | A deterministic record that *this* subject may take *this* action on *this* target, up to *this* much, in *this* round. Issued from what the person supplied and the authoritative record, never from model output. Scopes a write to a record the person established; it does not judge whether they asked for it, which is language work the model does. |
| **Effect Verification** | Reading the changed thing back after a side effect and reconciling it against what the action claimed. "The executor returned success" and "the money moved" are different claims; an unverified result never becomes authoritative. |
| **AgentHost** | How an agent reaches the operator shell — title, commands, stage labels, session rows, and how to build it. Named in the agent's own config (`agent.host`) and resolved, never imported, so `agent_platform` names no agent. |

## 3. Design process (mandatory order — do not skip ahead)

1. **Ubiquitous language first.** Before touching a module not already
   covered by §2, extend the glossary and confirm terms with Karan before
   writing an interface.
2. **Deep-module design.** Default to fewer, well-hidden abstractions.
   Before adding a new public method to any interface in `LLD.md`, ask:
   "does this leak an implementation detail the caller shouldn't need to
   know?" If yes, redesign rather than expose it.
3. **TDD as the coding discipline**, decisions resolved explicitly before
   implementation begins, per module:
   - **Unit granularity**: one test class per public method group, one
     assertion-focus per test (no "test everything in one giant test").
   - **What gets mocked**: only true I/O boundaries (LLM HTTP calls, DB
     connections, filesystem, external APIs). Never mock a sibling
     `agent_platform` module in a unit test that exists to prove the
     integration between them — that belongs in `tests/integration/`.
   - **Which behaviors are under test**: derived directly from the
     interface's docstring contract in `LLD.md`, not from the
     implementation once it exists (write the test from the interface,
     then the implementation to satisfy it).
4. **Interview-style planning.** When starting a new module not yet
   detailed in `LLD.md` (e.g. a future episode's subsystem), resolve one
   design decision at a time with Karan — branch by branch — rather than
   presenting a finished plan. Ask before building anything non-trivial.
   This applies to the coding agent as much as to a chat-based design
   session: if `LLD.md` doesn't already specify an interface, stop and
   ask rather than inventing one silently.

## 4. Build phasing (mapped to episodes and modules)

Each phase ends with: `ruff` clean, `mypy --strict` clean, `pytest`
coverage ≥ 80% on everything touched, and — where applicable — the
episode's own exit artifact demonstrably working end-to-end by running
the agent under that episode's config overlay, visible in the TUI
(ADR 0006 removed the per-episode scripts).

| Phase | Scope | Episodes | Definition of done |
|---|---|---|---|
| **0 — Walking skeleton** *(current)* | Every module/file exists; stub methods self-log (timestamp + per-subsystem color) on call | — | Full call graph traceable live via TUI/logs before any real logic lands |
| **1 — Platform core** | `config`, `logging`, `llm`, `persistence`, `tui` **live status feed** (spinner, human-readable stage labels, token counters, copy — no DB/memory/audit panels yet) | — | Config toggles work live, no restart; feed shows a started/completed pair for every stage in a canned request |
| **2 — customer_support_agent v1** | `core/graph.py` (LangGraph), `tools`, `repositories`, minimal `gatekeeper` (E01-E02) | E01-E04 | Structuring attack (E02) and the WF-vs-AG triage (E03-E04) both pass as regression tests |
| **3 — Pattern library shape** | `design_patterns/` for WF/AG patterns at pattern-shape level | E05-E07 | Each pattern's defining-property test passes independently of `customer_support_agent` |
| **4 — Harness contract** | `context` (5-stage pipeline only, not yet RAG), `evals` (Eval Pyramid + NOLLM cascade) | E08-E11 | Eval manifest checked in; NOLLM cascade is the one implementation four future callers will depend on |
| **5 — Gatekeeper hardening** | Full `gatekeeper` (Surface Classes, Tier-1/2, ledger), `security/auth` (RBAC/PoLP) | E09, E10(v6), E12 | Four E12 live attacks pass as permanent regression tests |
| **6 — Memory** | `memory` (all 4 stores), write pipeline, poisoning defenses, `context` RAG/consolidation | E13-E19 | E16 false-recall test passes (Memory Inspector TUI panel itself remains deferred — see Phase 1b below) |
| **7 — Control loop** | `agent_platform/control_loop` built for real (unparks it): `AgentFSM` + `LoopControllerImpl` merged package; `customer_support_agent/control_loop/` migrates off direct LangGraph-checkpointer wiring onto this package | E05, E20, E23 | Six induced-loop fixtures each aborted at the correct layer; crash-recovery restores a legal state without duplicating a side effect |
| **8 — Prompt compilation** | `agent_platform/prompt_compilation` built for real (unparks it) | E21 | One production prompt re-implemented as a compiled module with a regression-tested eval score |
| **9 — Tool registry at scale** | `tools` retrieval-based selection | E22 | 50-tool degradation/recovery demo passes numerically |
| **10 — Output guarantees** | `guardrails` (Pydantic + Outlines/LMQL) | E24, E28-E29 | Constrained-decoding path shows flat latency vs. retry-loop tail latency, measured |
| **11 — Hallucination protection** | `hallucination_check` | E25-E27 | Grounding-vs-reasoning diagnoser correctly routes both fixtures |
| **12 — Security** | `security/pii`, `security/prompt_injection`, `security/content_moderation` | E30-E33 | E31's OWASP control-mapping table passes as an actual test suite |
| **13 — Sandbox execution** | `agent_platform/sandbox` built for real (unparks it) — Docker-container tier only | E34 | `docker_container` tier exercised against a real container round-trip |
| **14 — Resilience** | `resilience` (all 7 patterns) | E35 | Bulkhead isolation and saga compensating-rollback both pass as regression tests |
| **15 — Observability & SRE** | `observability` (tracing, SLOs, cost-stack aggregation, queue math), `audit` | E36-E39 | Full trace reconstructs causal order; Little's Law worked example passes; E37 scores the harness's own maturity from real evidence |
| **15b — TUI panels** *(next-phase TODO, timing flexible relative to 16-18)* | `DbBrowserPanel`, `MemoryInspectorPanel`, `AuditTrailPanel`, `FeatureTogglePanel` | E01 onward (deferred) | All four panels populate against a real seeded DB + completed trace |
| **16 — Multi-agent** | `agent_platform/multi_agent` — **design starts here**, not before: run the interview-style process (§3) to establish ubiquitous language and interfaces before any implementation | E40-E43 | Series-reliability calculator matches E40's table; live-refactor-and-remove test passes |
| **17 — Capstone hardening** | Full attack suite across all subsystems; E44-E46 economics analysis delivered as episode content/reporting over Phase 15's cost-stack data, not new code | E44-E48c | Every attack listed in E48b passes; system demonstrably survives model/tool/policy/schema/context-window/traffic changes (E48c) |

Phases 3-18 are not strictly serial where dependency graphs (HLD §8)
allow parallel work, but a phase's gate must pass before anything in a
later phase is allowed to depend on it.

## 5. CI/CD

`.github/workflows/ci.yml` (currently missing — Phase 1 blocker, flagged
as a known gap):

```yaml
name: CI
on: [push, pull_request]
jobs:
  quality:
    steps:
      - run: uv sync
      - run: ruff check .
      - run: mypy --strict src/
      - run: pytest --cov=src --cov-fail-under=80 --cov-report=term-missing
```

- No LLM-endpoint-dependent test runs by default; those are marked and
  skipped unless an explicit env flag + credentials are present (keeps CI
  deterministic and free).
- A merge is blocked if coverage drops below 80% **on files touched by
  the PR**, not just the repo average — prevents coverage dilution.

## 6. Coding standards checklist (apply to every PR / generated module)

- [ ] Methods ~5-20 lines; if a method exceeds ~20 lines, extract.
- [ ] One concern per module; no module imports a concern it doesn't own
      (check against HLD §8's layering).
- [ ] Every swappable component (LLM provider, memory backend, DB,
      sandbox tier) is behind an ABC/Protocol; callers receive it via
      constructor injection, never instantiate a concrete class directly.
- [ ] No hardcoded values or magic numbers — every constant traces to a
      `config` field.
- [ ] No CWD-relative paths — package-relative (`importlib.resources` or
      equivalent) only.
- [ ] No orphaned files — every module is imported from somewhere real
      (checked via a dead-code scan, not just "looks referenced").
- [ ] Every declared `Protocol`/`ABC` has at least one concrete
      implementation verified by `mypy --strict` (this was a caught
      regression: a `Protocol` with no real implementer).
- [ ] Runtime artifacts (`.coverage`, `.pytest_cache`, `data/*.db`,
      `logs/`) are gitignored, never committed.
- [ ] `.env.example` exists and matches every `config` field the README
      references (this was a caught regression: README referenced a
      nonexistent file).
- [ ] LLD patterns (strategy/adapter/factory/builder/observer/facade/
      decorator) are applied where `LLD.md` names them for that module —
      not sprinkled elsewhere for their own sake.

## 7. Known failure patterns (from prior remediation passes — do not repeat)

- Episode code silently duplicating `agent_platform` (own LLM client,
  own auth checks, print-based logging) instead of composing it.
- Orphaned/dead files left behind after refactors.
- A Protocol declared but not actually implemented by its concrete
  classes.
- CWD-relative file paths that break outside repo-root invocation.
- Config fields declared with no corresponding adapter built.
- Runtime artifacts committed to git.
- Claiming work is "done" from reading code alone — the verification
  standard is `ruff` + `mypy --strict` + `pytest` (with coverage) run for
  real, not "looks correct."

## 8. Testing strategy summary

| Level | Scope | Owner | Cadence |
|---|---|---|---|
| Unit | One module, I/O boundaries mocked | every module (§1 mirror rule) | every PR |
| Integration | Two or more real modules composed, no mocking between them | cross-subsystem invariants (`LLD.md` §23) + `customer_support_agent` | every PR |
| Eval (component/integration/golden-task) | Behavioral correctness against the Eval Pyramid | `evals` | component+integration on every PR; golden-task nightly |
| Eval (production shadow) | Live-traffic shadow scoring | `evals` + `observability` | continuous against the running agent |
| Adversarial/attack-path | Security and guardrail invariants | `gatekeeper`, `guardrails`, `security` | every PR (fast subset) + full suite at each phase gate |

## 9. Resolved decisions (see `PRD.md` §11 for the full list) and what's still open

Episode numbering (48-episode v10), new-package placement (`sandbox`,
`control_loop`, `multi_agent`, `prompt_compilation` as parked/stub
`agent_platform/` packages), the removal of `economics` as a package,
and current-phase TUI scope are all resolved — see `PRD.md` §11 and
this document's §1/§4. One item remains genuinely open and should not
be silently defaulted:

1. **Local LLM adapter**: config already declares `local_model`/
   `local_base_url` fields — confirm before Phase 1 closes whether the
   `LocalLlamaCppClient` adapter ships in Phase 1 alongside those fields
   or is deferred to a later phase (shipping config ahead of the
   adapter was a previously caught regression; don't repeat it either
   way — if deferred, the fields themselves should be deferred too,
   not left dangling).