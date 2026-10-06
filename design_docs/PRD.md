# PRD — Agentic AI Systems Platform (`agentic-harness-core`)

**Status:** Draft v1 — build spec for coding-agent execution
**Scope:** Whole platform — `agent_platform` shared library, `customer_support_agent`
running project and `research_agent` (the per-episode demos and the WF/AG reference
library were retired by ADR 0006) — as required to support all 48 episodes of *The Agentic AI Systems
Masterclass*.
**Companion documents:** `HLD.md` (architecture), `LLD.md` (subsystem-level design),
`Program_Design.md` (repo layout, build phasing, standards).

---

## 1. Vision

`agentic-harness-core` is a production-grade reference implementation of a
governed agentic AI system. It is simultaneously:

1. **The running example** for a 48-episode, architect/CTO-level video series —
   every episode's "Exit Artifact" is a real, working, tested piece of this
   codebase, not a slide.
2. **A standalone product**: a LangGraph-based customer-support refund agent
   (`customer_support_agent`) that demonstrates every governance discipline
   the series teaches, operating against a real (seeded) policy and order
   dataset.
3. ~~**A teaching library** (`design_patterns/`)~~ — retired by ADR 0006; patterns are taught through the agents. Originally: independent, minimal reference
   implementations of the 15 WF/AG patterns and 5 cross-cutting concept
   patterns named in Module 1, decoupled from the running example so each
   pattern can be understood in isolation.

The codebase is a marketing and credibility asset: it must read as code
written to the standard of Anthropic, OpenAI, or Google, because it is the
proof the series' architectural claims are real, not aspirational.

## 2. Problem statement

Most public "agent" material is either (a) a toy notebook with no
authority model, no resilience, no audit trail — or (b) a wall of
architecture-diagram slideware with no working code behind it. There is no
widely available reference that shows, in real running code, how a model
becomes safe to deploy: bounded authority, deterministic gatekeeping,
memory with provenance, loop termination guarantees, structured-output
guarantees, hallucination measurement, and full audit lineage — all
composed into one coherent, testable system.

## 3. Users / personas

| Persona | Need |
|---|---|
| **Series viewer — Principal/Staff/Architect** | Wants to see the exact failure mode, then the exact architectural fix, in running code they could adapt at work. |
| **Series viewer — CTO/CXO** | Wants defensible governance claims ("show me the ledger, show me the rule for who can change the ceiling") they can repeat to their own board/regulators. |
| **Karan (author/maintainer)** | Needs the codebase itself to be premium-quality, coherent across 48 episodes with no continuity breaks, and buildable in phases by a coding agent from these documents. |
| **Coding agent (Claude Code, or similar)** | Needs an unambiguous, phase-ordered build spec — this PRD plus HLD/LLD/Program_Design — that it can execute without re-deriving architectural decisions already made. |

## 4. Goals

- **G1 — Governed autonomy, provably.** Every consequential action passes
  through a single, always-invoked Gatekeeper; no execution path bypasses it.
- **G2 — Full CoALA memory, never stubbed.** Working, episodic, semantic, and
  procedural memory are real, inspectable (via the TUI Memory Inspector),
  and each has its own write-path authorization and provenance model.
- **G3 — Composable, deep modules.** Every concern (logging, LLM adapters,
  guardrails, memory, security, resilience, tools, context, config) is its
  own module with a narrow interface; `customer_support_agent` *composes*
  `agent_platform`, never duplicates it.
- **G4 — Full audit lineage.** Every consequential decision is
  attributable to a versioned (model, policy, data, authority) tuple —
  regulator-ready, not an afterthought.
- **G5 — Measured, not asserted, maturity.** The system's own CMMI-style
  maturity level is scored from real SLO/audit evidence (E37), not
  self-declared (E04's rule, closed in E37).
- **G6 — Independently toggleable governance features.** Every safety
  feature (and each of its sub-features) is independently on/off
  configurable, so its marginal effect can be demonstrated and evaluated
  in isolation.
- **G7 — Per-episode demonstrability.** At every episode boundary, the
  repo is in a runnable state that supports that episode's cold-open and
  code-walkthrough.

## 5. Non-goals (v1)

- Multi-tenant SaaS hosting, billing, or a public-facing web UI (the TUI is
  the only interactive surface for v1).
- Support for LLM providers beyond NVIDIA NIM (cloud) and a local
  llama.cpp-style `mistral-nemo-12b` server.
- A general-purpose agent framework product — the platform is scoped to
  what the 48 episodes need, not a competitor to LangGraph/DeepAgents.
- Kubernetes/cloud-native deployment automation (E34/E39's isolation and
  versioning *disciplines* are implemented; a full CD pipeline to a cloud
  target is out of scope for v1).

## 6. Ubiquitous language (summary — full glossary in `Program_Design.md`)

Agent = Model + Harness · Authority ≠ Agency · Authority Hierarchy ·
Sandbox → Gatekeeper → Execution · Surface Classes (Locked / Editable /
Append-only / Human-controlled) · Tier-1 (deterministic, blocking) vs.
Tier-2 (semantic, advisory) · Automation Level (Sheridan 1–10) · Three
Gates (Path Cardinality, Blast Radius, Certifiability) · Working /
Episodic / Semantic / Procedural memory · Context Assembly pipeline ·
NOLLM Scoring Cascade · Eval Pyramid · Loop Controller · FSM state table ·
Circuit Breaker / Bulkhead / Saga / Idempotency · PII Filter · Trace ·
SLO / Error Budget · RBAC / Principle of Least Privilege.

## 7. Functional requirements

Requirements are grouped by the six Harness Contract subsystems (E08) plus
cross-cutting disciplines, each mapped to the owning module(s) and the
episode(s) that specify its behavior. Full acceptance detail lives in
`LLD.md`; this table is the traceability spine a coding agent should check
before marking a module "done."

### 7.1 Context assembly & memory

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-1 | Five-stage context pipeline: gather_candidates → score_relevance → fit_to_budget → format → order | `context` | E08 |
| FR-2 | Four CoALA memory stores, each independently testable, behind a shared `MemoryStore` interface | `memory/{working,episodic,semantic,procedural}` | E13–E14 |
| FR-3 | Memory write pipeline: extract → classify → provenance → conflict detection → confidence → authorization → route (auto-commit / curator queue / discard) | `memory` (write path) | E15 |
| FR-4 | Memory poisoning defenses: provenance tagging, decay-if-unconfirmed, hard delete/revocation | `memory`, `security` | E16 |
| FR-5 | Context failure detection (rot/distraction/confusion/clash/poisoning) measurable via synthesis-based eval at 10K–150K tokens | `evals`, `context` | E17 |
| FR-6 | Six context-management strategies: offload, reduce, retrieve, isolate, cache, consolidate | `context`, `memory` | E18 |
| FR-7 | Production RAG: semantic chunking, hybrid (lexical+dense) search, cross-encoder reranking, GraphRAG option | `context` (retrieval), `tools` | E19 |

### 7.2 Output parsing

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-8 | Structured-output validation via Pydantic (reject / auto-repair) | `guardrails` | E24 |
| FR-9 | Constrained decoding via Outlines/LMQL for the flat-latency, structurally-guaranteed path | `guardrails` | E24, E28 |
| FR-10 | No malformed model output may reach a side-effecting executor (invariant, tested) | `guardrails`, `gatekeeper` | E24 |

### 7.3 Tool execution

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-11 | Typed `ToolCall` interface, composed by every pattern | `tools` | E01, E05 |
| FR-12 | Tool registry with namespacing and retrieval-based selection (NOLLM cascade reused) | `tools` | E22 |
| FR-13 | Docker-container execution isolation, selected by blast-radius budget (full process/OS/container/microVM/remote ladder documented; only the Docker tier is in scope for implementation) | `agent_platform/sandbox` — **parked, TODO** | E34 |

### 7.4 Control loop

`agent_platform/control_loop` merges what earlier planning treated as
two subsystems (`loop_engineering` and `state_management`) into one
package — both govern the same trajectory object, and both are
**parked/TODO** this phase. `customer_support_agent/control_loop/`
composes this package (thin wiring only) rather than duplicating it;
until the package is built, that folder wires LangGraph's own
checkpointer directly as a tracked interim state.

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-14 | ReAct loop (`react_step()`) as the atomic trajectory unit | `agent_platform/control_loop` — **parked, TODO** | E05 |
| FR-15 | `LoopController` with six-layer watchdog stack (step count, wall-clock, token budget, cost budget, duplicate-action, semantic-stagnation) | `agent_platform/control_loop` — **parked, TODO** | E20 |
| FR-16 | Planner + Reflector + Evaluator composition, each independently swappable | `customer_support_agent/control_loop`, `evals` | E06 |
| FR-17 | Orchestrator-subagent decomposition with bounded subagent context | `agent_platform/multi_agent` — **stub only, design parked** | E07, E17, E40–E43 |

### 7.5 State management

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-18 | Explicit FSM: IDLE → PLANNING → ACTING → WAITING_TOOL → WAITING_HUMAN → DONE → FAILED, illegal transitions rejected | `agent_platform/control_loop` — **parked, TODO** | E23 |
| FR-19 | Checkpoint = `(state, transition history, durable facts)`, replayable after crash without duplicating side effects | `agent_platform/control_loop`, `persistence` — **parked, TODO** | E23 |
| FR-20 | Circuit Breaker modeled as its own FSM (Closed → Open → Half-Open), reusing E23's state-table discipline | `resilience/circuit_breaker` | E23, E35 |

### 7.6 Error handling / resilience

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-21 | Retry only paired with idempotency keys; capped backoff | `resilience/retry`, `resilience/idempotency` | E35 |
| FR-22 | Circuit Breaker per dependency | `resilience/circuit_breaker` | E35 |
| FR-23 | Bulkhead resource-pool isolation per dependency | `resilience/bulkhead` | E35 |
| FR-24 | Saga with defined compensating actions for the refund transaction chain | `resilience/saga` | E35 |
| FR-25 | Pub-Sub for async event-driven coordination (multi-agent, E42) | `resilience/pub_sub` | E12, E42 |

### 7.7 Authority, security & governance

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-26 | Authority Hierarchy (provider > org > product > workspace > user task > runtime reminders > tool observations > untrusted content); only ranks 1–5 can grant/expand authority | `gatekeeper` | E02 |
| FR-27 | Cumulative-spend ledger defeating structuring attacks | `gatekeeper` | E02 |
| FR-28 | Four Surface Classes (Locked/Editable/Append-only/Human-controlled); no execution path bypasses the Gatekeeper | `gatekeeper` | E12 |
| FR-29 | Tier-1 (deterministic, blocking) vs. Tier-2 (semantic, advisory-only) enforced structurally — Tier-2 can never gain veto power | `gatekeeper` | E12 |
| FR-30 | RBAC (NIST model) + Principle of Least Privilege, just-in-time scoped elevation | `security/auth` | E10 (v6 numbering; see episode doc for current slot) |
| FR-31 | Prompt-injection defense-in-depth: instruction/data separation, tool scoping, Gatekeeper as final structural backstop | `security/prompt_injection` | E30, E35 (six-layer) |
| FR-32 | PII detection, redaction (mask/tokenize/remove), minimization at input/memory/log surfaces | `security/pii` | E32 |
| FR-33 | Data lineage: every consequential decision attributable to a versioned (model, policy, data, authority) tuple | `audit`, `observability` | E33 |
| FR-34 | Automation-level policy table per action class (Sheridan 1–10, driven by Three Gates score) | `gatekeeper`, config | E03 |

### 7.8 Evals, hallucination, guardrails

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-35 | Four-level Eval Pyramid: component / integration / E2E golden-task / production shadow | `evals` | E09 |
| FR-36 | NOLLM Scoring Cascade: exact assertion → deterministic rule → embedding similarity → NLI entailment → calibrated LLM judge → human adjudication, with `nollm_scorer_ratio` tracked | `evals` | E10 |
| FR-37 | Eval integrity: poisoned-label detection, pairwise judge comparison, Cohen's kappa calibration | `evals` | E11 |
| FR-38 | Hallucination detection ladder (confidence → consistency/entropy → NLI → judge → human), claim-level verification | `hallucination_protection` | E25 |
| FR-39 | Grounding-failure vs. reasoning-failure differential diagnosis | `hallucination_protection` | E26 |
| FR-40 | Hallucination SLO / error budget, reviewer-calibration tracking (automation bias & algorithm aversion) | `hallucination_protection`, `observability` | E27 |
| FR-41 | Generate-under-constraint vs. generate-validate-retry, both available, latency difference measured | `guardrails` | E28 |
| FR-42 | Guardrail system tested via chaos injection, property-based tests, adversarial attack-path tests | `guardrails`, tests | E29 |

### 7.9 Observability & operations

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-43 | Every hop traced: trace ID, actor, authority context, tool, policy version, model version, state transition, result, latency, cost | `observability` | E36 |
| FR-44 | Traces pass through `PIIFilter` before persistence | `observability`, `security/pii` | E32, E36 |
| FR-45 | Agent SLOs tracked: task success, unauthorized-action rate, hallucination rate, escalation rate, loop-abort rate, tool failure rate, cost/success, recovery success, policy-violation rate | `observability` | E37 |
| FR-46 | Postmortem template enforced on every real incident; becomes a regression eval | `evals`, `observability` | E37 |
| FR-47 | Human-review queue capacity computed via Little's Law from escalation rate × review time | `observability` (queue module) | E38 |
| FR-48 | Independent versioning of prompt / tool schema / policy / evaluator / model / retrieval index / memory schema / guardrail / PII rules; canary rollout; no hard cutover on the Gatekeeper's live authority path | `Program_Design` (deployment), `config` | E39 |

### 7.10 Multi-agent & economics

Economics (E44–E46) is treated as a generic, largely narrative
discipline for this build — worked-example analysis over data
`observability` already emits, not a distinct engineering surface. No
`economics` package exists; FR-52/FR-53 below are `observability`/`llm`
requirements, not a separate module's.

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-49 | Series-reliability compounding calculator for agent chains (`P(success) = Π P_i`) | `agent_platform/multi_agent` — **stub only, design parked** | E40 |
| FR-50 | Configurable topology: supervisor / hierarchy / peer network / swarm | `agent_platform/multi_agent` — **stub only, design parked** | E41 |
| FR-51 | MCP (agent-to-tool) and A2A (agent-to-agent) protocol support with correlation IDs and idempotent delivery | `agent_platform/multi_agent`, `resilience/pub_sub` — **stub only, design parked** | E42 |
| FR-52 | Cost stack tracked per request: input/output tokens, tool calls, retrieval, evaluator calls, human review, orchestration, infra | `observability` | E44 |
| FR-53 | Model cascading (cheapest sufficient tier first) reusing the NOLLM cascade discipline | `llm` | E44 |

### 7.11 Prompt engineering

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-54 | DSPy signature/module/optimizer compilation path as an alternative to hand-written prompt strings, versioned as a procedural-memory artifact | `agent_platform/prompt_compilation` — **parked, TODO** | E21 |

### 7.12 Platform / developer experience

| Req | Description | Module | Episodes |
|---|---|---|---|
| FR-55 | Textual TUI live status feed (Claude-Code-style): every significant harness stage a request passes through — not just the final response — appears as a human-readable line (e.g. "Checking for prompt injection…", "Calling model…", "Checking refund eligibility…") with a spinner while in flight and a completion mark when done | `tui` | E01 onward |
| FR-55a | Running token counters visible in the TUI: session total and per-turn, both simultaneously | `tui` | — |
| FR-55b | Text-selection copy in the TUI (select a portion of visible text, copy it) — same interaction model as Claude Code's terminal UI | `tui` | — |
| FR-55c | DB browser, memory inspector (all 4 CoALA stores, real content), and audit-trail view — **deferred to next phase**; interfaces are specified in `LLD.md` so the module boundary is stable, but no panel implementation ships this phase | `tui` — **deferred, TODO** | E01 onward |
| FR-56 | File logging required independently of the TUI: the TUI's live feed is a human-readable summary only — full structured/debug detail for the same event goes only to the corresponding log file | `logging`, `tui` | — |
| FR-57 | Every governance feature (and sub-feature) independently toggleable on/off, live, without restart (Tier-2 mutable in-memory config singleton) | `config` | — |
| FR-58 | **Retired** ([ADR 0006](adr/0006-retire-episodes-and-pattern-library.md)): episodes are demonstrated by running `customer_support_agent` / `research_agent` under config overlays; there are no per-episode scripts | — | all |
| FR-59 | **Retired** ([ADR 0006](adr/0006-retire-episodes-and-pattern-library.md)): no separate pattern library; each agent's `design_docs/patterns.md` maps the patterns to where it uses them | — | M1, M4–M18 |

## 8. Non-functional requirements

| Category | Requirement |
|---|---|
| **Modularity** | Methods ~5–20 lines; one concern per module; deep-module interfaces (Ousterhout). |
| **Extensibility** | LLM provider, memory backend, and DB are swappable via ABC/Protocol + dependency injection; no concrete class hardcoded into a caller. |
| **Testability** | Unit + integration tests per module, CI-runnable; **80%+ coverage gate**, no exceptions without explicit sign-off. |
| **Observability** | No side-effecting action without a trace; no trace without a PII pass. |
| **Auditability** | Every consequential decision reconstructable from (model, policy, data, authority) versions. |
| **Config hygiene** | No hardcoded values or magic numbers; all values read from configuration; `SECTION__FIELD` pydantic-settings convention except bare provider API keys. |
| **Security posture** | Zero Trust — no caller is trusted merely for being internal/prior-approved; Complete Mediation — no cached "yes." |
| **Resilience** | Retries always idempotent; one dependency's failure cannot exhaust another's budget; every partial transaction has a compensating action. |
| **Performance** | Constrained-decoding path must show flat latency vs. retry-loop's tail latency (measured, E28). |
| **Repo hygiene** | `.gitignore` covers `.coverage`, `.pytest_cache`, `data/*.db`, `logs/`; no CWD-relative paths — package-relative only. |

## 9. Success metrics

- 100% of the FR table above has a passing unit/integration test and appears
  in at least one episode's exit artifact.
- Coverage ≥ 80% on every merged module; 0 mypy --strict errors; ruff clean.
- Every governance feature demonstrably changes measured behavior when
  toggled off in an eval run (proves G6 isn't cosmetic).
- The harness's own CMMI-style maturity level (E04/E37) is backed by real
  SLO/audit evidence, not asserted.
- Zero orphaned/dead files, zero Protocol-without-implementation findings,
  zero committed runtime artifacts (the exact regressions already caught
  once — see `Program_Design.md` §"Known failure patterns").

## 10. Phased roadmap (summary — full detail in `Program_Design.md`)

**Phase 0 — Walking skeleton** *(current state)*: every module/file exists
with stub methods that self-log on call, proving config + call ordering
end-to-end via the TUI before any real logic lands. → **Phase 1 — Platform
core**: logging, TUI **live status feed** (spinner, human-readable stage
labels, token counters, text-selection copy — DB browser/memory
inspector/audit view explicitly deferred), config, LLM adapters,
persistence built for real. → **Phase 2 — customer_support_agent v1**:
LangGraph graph, tools, repositories, real Gatekeeper + Authority
Hierarchy + ledger. → **Phase 3+**: one phase per remaining module
family (memory → guardrails → hallucination protection → `control_loop`
→ resilience → security → observability/SRE → `sandbox` →
`prompt_compilation` → `multi_agent`), each phase gated by its own
episodes' exit artifacts and the 80% coverage bar. Full phase-by-phase
detail (including which modules are parked vs. stub-only vs. live in
which phase) is `Program_Design.md` §4.

## 11. Resolved decisions (previously open questions)

- **Episode numbering**: the 48-episode v10 curriculum is the reference
  for episode numbering (E01–E48); since ADR 0006 an episode is a config
  overlay over an agent, not an `episodes/` folder.
- **New-package placement**: `sandbox`, `control_loop` (merging the
  earlier `loop_engineering`/`state_management` split),
  `multi_agent`, and `prompt_compilation` all land as top-level
  `agent_platform/` packages — **parked (interfaces defined, no
  implementation) this phase**, except `multi_agent`, which is a
  **stub only** (folder exists, design not yet started).
  `customer_support_agent/control_loop/` composes
  `agent_platform/control_loop` (thin wiring) rather than
  duplicating its logic.
- **Economics**: no `agent_platform/economics` package. Cost-stack
  aggregation is a normal part of `observability`; TCO/ROI/Amdahl
  analysis (E44–E46) is treated as narrative/analytical content over
  data `observability` already emits, not a distinct code surface.
- **TUI scope, current phase**: a single live status feed (spinner +
  human-readable per-stage labels + running token counters +
  text-selection copy), functioning as a summary surface only — full
  detail stays in the per-subsystem log files. DB browser, memory
  inspector, and audit-trail view are designed at interface level
  (`LLD.md` §20) but not implemented until next phase.