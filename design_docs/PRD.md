# PRD — Agentic Harness Platform (`agentic-harness-core`)

**Status:** current as of 2026-10-12. Requirements carry their build status; decisions behind
them are in [ADR.md](ADR.md).
**Companion documents:** [HLD.md](HLD.md) (architecture), [LLD.md](LLD.md) (components),
[Program_Design.md](Program_Design.md) (standards, layout, process).

## 1. Vision

`agentic-harness-core` is a production-grade reference implementation of governed agentic AI
systems. It is at once:

1. **The running example** for *The Agentic AI Systems Masterclass*, a 48+ episode series for
   CTO/CXO and Principal/Staff/Architect audiences. An episode runs one of the agents under the
   configuration it teaches; what is on screen is the code that runs.
2. **A shared platform** (`agent_platform`) that any agent composes: model access, the harness
   controls, and the always-on plumbing — written to the standard of a frontier lab, not
   tutorial scaffolding.
3. **Three agents on that platform**, each demonstrating a different shape of agent:
   - **Customer Support** — a governed, money-moving tool-use agent (refunds, cancellations).
   - **Research** — a scoped, contract-bound, independently audited research agent with a
     direct-answer path.
   - **Ops Analyst** — an agent whose only tools arrive over MCP, answering questions about how
     the other agents are running.

## 2. Problem

Most public agent material is either a notebook with no authority model, resilience or audit
trail, or architecture slideware with no running code. There is no widely available reference
showing, in one coherent tested system, how a model becomes safe to deploy: bounded authority,
deterministic mediation, memory with provenance, loop termination, structured output,
hallucination measurement, human review and full audit lineage.

## 3. Users

| Persona | Need |
|---|---|
| Series viewer — Principal / Staff / Architect | The exact failure mode, then the exact architectural fix, in code they could adapt |
| Series viewer — CTO / CXO | Defensible governance claims ("show me the ledger, show me who can change the limit") |
| Maintainer | A codebase coherent across every episode, buildable in phases, verifiable by its gates |
| Coding agent | An unambiguous spec (these documents and `CLAUDE.md`) it can execute without re-deriving decisions |
| An engineer building a new agent | A platform that gives a fourth agent screening, grounding, memory, review, analytics and evals by composition and configuration alone |

## 4. Goals

- **G1 — Governed autonomy, demonstrably.** *Retired as written* (ADR P-10). Consequential
  actions pass the Gatekeeper when it is declared; it may be withheld, and then calls still run
  through the Executor and are audited as UNGOVERNED — so its effect can be shown.
- **G2 — Full CoALA memory, never stubbed.** Working, episodic, semantic and procedural memory
  are real, inspectable and governed.
- **G3 — Composable, deep modules.** Agents compose `agent_platform`; the platform names no agent.
- **G4 — Full audit lineage.** Every consequential decision is attributable to versioned
  (model, policy, data, prompt, tool schema) stamps and a standing.
- **G5 — Measured, not asserted.** SLOs, error budgets and eval scores come from recorded facts.
- **G6 — Every harness control independently toggleable**, live, so its marginal effect can be
  demonstrated and evaluated.
- **G7 — Demonstrable at every episode boundary**, by configuration, not by separate scripts.
- **G8 — A new agent gets the platform for free**: common metrics, review, memory, evals and the
  console by implementing a host, a plugin and a config overlay.

## 5. Non-goals

- Multi-tenant SaaS hosting, billing, or a web UI (the Textual console is the interactive surface).
- Providers beyond NVIDIA NIM and a local OpenAI-compatible server.
- A general-purpose agent framework competing with LangGraph.
- Cloud deployment automation (versioning and isolation disciplines are taught; no CD pipeline).

## 6. Parked and deferred

Recorded so nothing is rebuilt by accident or claimed as built.

| Item | Episodes | State |
|---|---|---|
| Docker execution sandbox (`sandbox`) | E34 | Parked; the empty package was deleted. Tools execute in-process. |
| Multi-agent topologies, A2A, reliability compounding (`multi_agent`) | E40–E43 | Parked; the empty package was deleted. MCP (agent-to-tool) is built. |
| DSPy prompt compilation (`prompt_compilation`) | E21 | Parked; the empty package was deleted. Skills are the versioned procedural artifact. |
| A genuine fixed loop | — | No implementation; pass^k trials are the leading candidate (ADR P-06). |
| Research agent eval suite | — | Removed with the fixture world (ADR R-10); to be rebuilt over live or recorded data. |
| Eval facts in the analytics source (`eval_score`) | — | Owed (ADR P-24). |
| NL-to-SQL golden set for the ops agent; MCP HTTP transport auth | — | Owed (ADR P-25). |
| Memory decay-if-unconfirmed | E16 | Not built; its unread config field was removed. |
| Gatekeeper kill switch, `/policy` commands, evidence-gated maturity levels | E04, E37 | Deferred (ADR P-09). |

## 7. Functional requirements

Status: **Built**, **Partial** (the mechanism exists; part of the requirement is open),
**Amended** (built differently, see the ADR), **Parked**, **Retired**.

### 7.1 Context and memory

| Req | Description | Module | Status |
|---|---|---|---|
| FR-1 | Five-stage context pipeline: gather → score → fit → format → order | `context` | Built |
| FR-2 | Four CoALA stores, each independently testable | `memory` | Built (files, ADR P-13) |
| FR-3 | Governed write pipeline: extract → classify → provenance → conflict → confidence → authorize → route | `memory` | Built |
| FR-4 | Memory-poisoning defences: provenance, decay-if-unconfirmed, revocation | `memory`, `security` | Partial (no decay) |
| FR-5 | Context-failure detection (rot, distraction, confusion, clash, poisoning) | `context` | Partial (detection built; long-context synthesis eval not built) |
| FR-6 | Six context strategies: offload, reduce, retrieve, isolate, cache, consolidate | `context` | Built |
| FR-7 | RAG: chunking, hybrid lexical + dense search, cross-encoder rerank, graph retrieval | `context`, `memory` | Built |
| FR-60 | Dreams: long-term memory written only by consolidation of dream events, for every agent | `memory/dream` | Built |
| FR-61 | Agent profile and skills as procedural memory, selected without a model call | `memory/procedural/skills` | Built |

### 7.2 Output

| Req | Description | Module | Status |
|---|---|---|---|
| FR-8 | Structured-output validation (reject / auto-repair) | `guardrails` | Built |
| FR-9 | Constrained decoding for the structurally guaranteed path | `guardrails`, `llm` | Partial (grammar adapter for a local server) |
| FR-10 | No malformed output reaches a side-effecting executor | `guardrails`, `tools` | Built |

### 7.3 Tools and execution

| Req | Description | Module | Status |
|---|---|---|---|
| FR-11 | Typed tool calls | `tools` | Built |
| FR-12 | Namespaced registry, retrieval-based selection, model names | `tools` | Built |
| FR-13 | Container execution isolation | `sandbox` | Parked |
| FR-62 | One tool path for every model-proposed call, recording the stage reached | `tools/path.py` | Built |
| FR-63 | Tools over MCP, governed by the deployment's envelope | `knowledge/mcp` | Built |

### 7.4 Control loop and state

| Req | Description | Module | Status |
|---|---|---|---|
| FR-14 | ReAct loop | research and ops agents | Built |
| FR-15 | Loop watchdog stack: steps, wall clock, tokens, cost, duplicate action, semantic stagnation, cycles, convergence | `loop_control` | Built (`LoopGovernor`) |
| FR-16 | Planner + reflector + evaluator, each swappable | agents | Built |
| FR-17 | Orchestrator–subagent decomposition with bounded subagent context | `durability/isolated.py`, research audit | Partial (isolated sub-graph; no multi-agent topology) |
| FR-18 | Explicit legal-transition state machine | `loop_control/state.py` | Built |
| FR-19 | Checkpoint and replay without duplicating side effects | `durability`, `resilience` | Built |
| FR-20 | Circuit breaker as its own state machine | `resilience` | Built |
| FR-64 | Stop / Resume / Abort for every durable agent | `durability/controls.py` | Built |

### 7.5 Resilience

| Req | Description | Module | Status |
|---|---|---|---|
| FR-21 | Retry only with idempotency keys; capped backoff | `resilience` | Built |
| FR-22 | Circuit breaker per dependency | `resilience` | Built |
| FR-23 | Bulkhead per dependency | `resilience` | Built |
| FR-24 | Saga with compensations for the refund chain | `resilience`, CSA | Built |
| FR-25 | Pub-sub for event-driven coordination | `resilience` | Built (in-memory) |
| FR-65 | Cross-process rate limit and monthly quotas per provider | `resilience/rate_limiter` | Built |
| FR-66 | Hedged backups per model; queue time not charged to the call | `llm/chain.py`, `resilience` | Built |

### 7.6 Authority, security, governance

| Req | Description | Module | Status |
|---|---|---|---|
| FR-26 | Authority hierarchy; only the top ranks may grant authority | `gatekeeper` | Built |
| FR-27 | Cumulative ledger against structuring | `gatekeeper` | Built (off for CSA) |
| FR-28 | Surface classes; no path bypasses the Gatekeeper | `gatekeeper` | Amended (G1 retired: UNGOVERNED is audited) |
| FR-29 | Tier-1 deterministic and blocking; Tier-2 advisory only | `gatekeeper` | Built |
| FR-30 | RBAC and least privilege | `security/auth` | Built |
| FR-31 | Prompt-injection defence in depth | `security`, research agent | Built |
| FR-32 | PII detection, redaction, tokenisation at the provider boundary, audited reveal | `security/pii` | Built |
| FR-33 | Lineage: decisions attributable to versioned stamps | `audit` | Built |
| FR-34 | Automation-level policy table per action class | `gatekeeper` | Retired (no consumer; config removed) |
| FR-67 | Standings on every decision (VOID, RATIFIABLE, VOIDABLE, VALID, UNGOVERNED) | `gatekeeper` | Built |
| FR-68 | One human-review queue, a comment on every decision | `review` | Built |

### 7.7 Evals, hallucination, guardrails

| Req | Description | Module | Status |
|---|---|---|---|
| FR-35 | Layered evals | `evals` | Amended (three graders: oracle, DeepEval, Promptfoo; ADR P-28) |
| FR-36 | NOLLM scoring cascade with `nollm_scorer_ratio` | `scoring` | Built (tier 5 declines until a calibrated judge is wired) |
| FR-37 | Judge calibration (Cohen's κ) | `judge` | Built |
| FR-38 | Hallucination detection ladder, claim level | `hallucination_check` | Built (levels, fact ledger, NLI) |
| FR-39 | Grounding vs reasoning failure diagnosis | `hallucination_check` | Built |
| FR-40 | Hallucination SLO and error budget | `observability`, `analytics` | Built |
| FR-41 | Generate-under-constraint vs generate-validate-retry, latency measured | `guardrails` | Partial |
| FR-42 | Guardrails tested by adversarial attack paths | tests, Promptfoo | Built (attack suites); chaos and property tests partial |

### 7.8 Observability and operations

| Req | Description | Module | Status |
|---|---|---|---|
| FR-43 | Every hop traced and recorded as facts | `observability` | Built |
| FR-44 | Traces PII-redacted before export | `observability`, `security` | Built |
| FR-45 | Agent SLOs and error budgets | `observability`, `analytics` | Built |
| FR-46 | Postmortem register | `observability` | Partial (no console command) |
| FR-47 | Review-queue capacity by Little's Law | `observability` | Built |
| FR-48 | Independent versioning of prompt, tool schema, policy, model, skills | `config`, `audit`, `memory` | Partial (stamps and skill versions; no canary rollout) |
| FR-69 | Analytics source: facts at platform choke points, Parquet export, metric catalogue | `analytics`, `observability` | Built |
| FR-70 | Analytics over MCP and generated dashboards | `knowledge/mcp`, `analytics` | Built |

### 7.9 Multi-agent and economics

| Req | Description | Module | Status |
|---|---|---|---|
| FR-49 | Series-reliability calculator for agent chains | — | Parked |
| FR-50 | Configurable multi-agent topology | — | Parked |
| FR-51 | MCP and A2A with correlation ids and idempotent delivery | `knowledge/mcp` | Partial (MCP built; A2A parked) |
| FR-52 | Cost per request (tokens, tool calls, judge calls) | `observability`, `analytics` | Built |
| FR-53 | Model cascading, cheapest sufficient tier first | `llm` | Built |

### 7.10 Prompting

| Req | Description | Module | Status |
|---|---|---|---|
| FR-54 | Compiled prompts (DSPy) as versioned artifacts | — | Parked |

### 7.11 Platform and developer experience

| Req | Description | Module | Status |
|---|---|---|---|
| FR-55 | Live stage feed in the console (human-readable cards, spinner, completion marks) | `interface/tui` | Built |
| FR-55a | Running token counters | `interface/tui`, `llm/metering.py` | Built |
| FR-55b | Text-selection copy | `interface/tui` | Built |
| FR-55c | Database, memory and audit inspection | `interface/commands` | Built as commands and the Review screen (`/memory`, `/audit`, `/review`, `/stats`) |
| FR-56 | File logging independent of the console | `logging` | Built |
| FR-57 | Every harness feature toggleable live without restart | `config`, `/harness` | Built |
| FR-58 | Per-episode scripts | — | Retired (ADR P-06) |
| FR-59 | Separate pattern library | — | Retired (ADR P-06); see [patterns.md](patterns.md) |

## 8. Non-functional requirements

| Category | Requirement |
|---|---|
| Modularity | Methods about 5–20 lines; one concern per module; deep modules |
| Extensibility | Providers, stores and databases behind Protocols and dependency injection |
| Testability | Unit and integration tests per module; 80% coverage gate; tests offline |
| Observability | No side-effecting action without a fact and an audit entry |
| Auditability | Every decision reconstructable from its trace and version stamps |
| Configuration | No hard-coded values; everything from configuration; `SECTION__FIELD` env convention |
| Security | Zero trust; complete mediation; PII never sent to a provider when declared |
| Resilience | Retries idempotent; one dependency cannot exhaust another's budget; every partial transaction compensated |
| Repository hygiene | Runtime state (`db/`, `data/`, `logs/`, `reports/`) and caches never versioned; no CWD-relative paths |

## 9. Success metrics

- `make verify` green: ruff, `mypy --strict`, pytest with ≥ 80% coverage, every composition root
  importable.
- Every harness feature demonstrably changes measured behaviour when withheld (the eval suites and
  `/harness` demos).
- Customer Support evals: oracle and Promptfoo at 100%; DeepEval above the soft gate.
- A new agent appears on the Agent Platform dashboard by running turns, with no analytics code.
- Zero orphaned modules, zero Protocol-without-implementation findings, zero committed runtime
  artifacts.
