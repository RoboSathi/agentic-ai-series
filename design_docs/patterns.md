# Patterns — where each agent uses them

The series teaches workflow/agentic (WF/AG) patterns and LLD/resilience patterns through the
running agents, not through a separate reference library (ADR P-06). This page maps each
pattern to where it lives in code. Paths are relative to `src/`. Edit it when the code moves.

## Coverage at a glance

| Pattern | Customer Support | Research | Ops |
|---|---|---|---|
| Sequence (fixed pipeline) | ✓ front-door stages | — | — |
| Branch | ✓ | ✓ | ✓ |
| Router | ✓ intent | ✓ research kind | — |
| Tool use | ✓ (not ReAct) | ✓ direct answer | ✓ over MCP |
| ReAct | — | ✓ worker, direct answer | ✓ |
| Planning | ✓ plan lookup | ✓ plan DAG | — |
| Fan-out / fan-in, map-reduce | — | ✓ | — |
| Reflection (Producer–Critic) | ✓ deterministic critic | ✓ evidence and report | — |
| Evaluator–optimizer | ✓ quality gate | ✓ audit → gap-fill / revise | — |
| Reflexion (learning across runs) | ✓ learned skills from refusals | ✓ learned skills from run findings | — |
| Orchestrator–subagent | — | ✓ isolated audit sub-graph | — |
| Human in the loop | ✓ review queue + interrupt | ✓ interview, `/approve` | — |
| RAG | ✓ knowledge base | — | — |
| Memory systems (CoALA) | ✓ all four | ✓ all four | ✓ procedural |
| Guardrails / grounding | ✓ NLI + fact ledger | ✓ citation guard, NLI | ✓ figure check |
| Fixed loop | — | — | — |

`fixed_loop` has no implementation in any agent: the research agent's budgeted loops are
stopped by the model or a stop rule, and the budget only caps them (ADR P-06).

## Customer Support agent

### Workflow and agentic

| Pattern | Where | Notes |
|---|---|---|
| sequence | `graph/edges.py` · pii_filtering → prompt_injection → content_moderation → deterministic_checks → classify_intent → assemble_context | Fixed order; a withheld stage is not built and its neighbours are wired together |
| branch | `graph/edges.py` · `_after_authenticate`, `_after_screening`, `_after_classify_intent`, `_after_finalize` | Deterministic predicates halt or continue the turn |
| router | `intent/router.py` · classify_intent → read_intent | Deterministic rejects, phrase rules, then a local classifier (Laya by default), then one clarification question; reads route with no model call |
| tool_use | `graph/nodes/propose.py` ⇄ `graph/nodes/execute.py` | Typed calls through the platform tool path, bounded by the loop governor; deliberately not ReAct |
| human_in_the_loop | `graph/nodes/review_wait.py` (await_approval, an interrupt) + `console/review_handler.py` | Above-limit refunds and cut replies become review items; approval resumes the durable run; a comment is required |
| rag_pipeline | `graph/builder.py` · `FiveStagePipeline` with the knowledge-base context source | |
| memory_systems | `graph/builder.py` · `MemoryFacade` over the four stores; `memory/profile.py` | Skills per intent selected by `intent:`/`route:` signals; learned skills from recurring refusals after curator approval |
| profile | `prompts/AGENT.md` | Identity and non-negotiables, always in the prompt |
| planning | `loop_control/quality_gate.py` · `PlanLookup` | Tool order per intent from config, committed before acting |
| reflection | `loop_control/quality_gate.py` · `GroundedCritic` | One deterministic critique of the draft against tool outputs |
| evaluator_optimizer | `loop_control/quality_gate.py` · `OutputQualityGate` | Critique, at most one repair, `IndependentScorer` |

### LLD and resilience

| Pattern | Where |
|---|---|
| saga | `resilience/refund_saga.py`, `resilience/cancellation_saga.py` |
| retry + idempotency + circuit breaker + timeout | the model chain (`agent_platform/llm/chain.py`) and the Executor |
| strategy | `loop_control/quality_gate.py` · `Planner`, `Reflector`, `Evaluator` protocols |
| factory | `graph/builder.py` (composition root) |
| facade | `agent_platform` `MemoryFacade`, `SecuritySuite` |
| adapter | `verification/refund_effects.py` · `RefundEffectSource` for the platform verifier |

## Research agent

### Workflow and agentic

| Pattern | Where | Notes |
|---|---|---|
| router | `scoping/kind.py` · research_kind node | Cue rules, the example classifier, then the LLM; the kind picks checklist, rubric, stop rule and report |
| human_in_the_loop | `graph/scoping_graph.py` · await_user (`durability/hitl.await_decision`) | The interview and rubric review pause the durable graph; `/approve` freezes the contract |
| planning | `planning/planner.py` + `planning/dag_validation.py` · plan_dag | A DAG of sub-questions, validated before anything runs |
| fan_out_fan_in | `graph/run_graph.py` · batch_start → research_node (`Send` per ready node) → batch_done | Each batch runs under the scheduler's cap and joins at the boundary |
| react | `research/worker.py` · research_node; `graph/answer_graph.py` · reason ⇄ act | Bounded act–observe loops |
| agentic_retrieval | `research/worker.py` + `research/reflector.py` | The agent chooses what to search next from what it has found |
| map_reduce | `research/extractor.py` (map) · `synthesis/synthesizer.py` (reduce) | Pages map to verbatim claims; synthesis reduces the ledger to the report |
| reflection (Producer–Critic) | `research/reflector.py`; `graph/run_nodes.py` · independent_audit → revise | Over the evidence (gaps → follow-ups) and over the report (audit findings → bounded revision); the critics are external checks |
| reflexion | `memory/run_events.py` → `make dream ra` → `/review` → plan | Findings recurring across runs of one kind become a learned skill a curator approves |
| evaluator_optimizer | `graph/run_nodes.py` · synthesize → independent_audit → gap_fill / revise | |
| orchestrator_subagent | `audit/independent.py` on `agent_platform` `IsolatedGraphRunner` | The audit sub-graph receives only `(contract, report, ledger_snapshot)` |
| memory_systems | `archive/store.py`, `memory/` | Episodic run index, semantic facts (source reliability), learned skills |

### LLD and resilience

| Pattern | Where |
|---|---|
| strategy | `research/stop_rules/` (one per kind) |
| adapter | `agent_platform/knowledge/web/providers.py` · `TavilyProvider`, `DuckDuckGoProvider` behind `SearchProvider` |
| chain of responsibility | `agent_platform/knowledge/web/router.py` · `SearchRouter` (free provider first, metered fallback) |
| factory | `graph/builder.py` (composition root) |
| builder | `scoping/brief.py` · `BriefBuilder` |
| observer | `StageReporter` + `AuditLedger` telemetry; the `WebGateway` observer |
| template method | `audit/independent.py` · the audit judge on `agent_platform.judge.LlmJudge` |
| facade | `agent_platform` `MemoryFacade`, used by `archive/store.py` |
| retry + idempotency + circuit breaker | `agent_platform/knowledge/web/gateway.py` · live search and fetch through `ResilientCaller` |
| bulkhead | `agent_platform/runtime/scheduling/scheduler.py` · per-batch concurrency cap |

## Ops agent

### Workflow and agentic

| Pattern | Where | Notes |
|---|---|---|
| react | `graph/graph.py` · propose ⇄ execute | Lists, describes, queries, and corrects its SQL from the server's refusals; bounded by `ops_agent.max_steps` |
| procedural memory | `prompts/AGENT.md` + `prompts/skills/analytics-investigation` | One `always` skill, versioned in `SkillStore` and audited per turn |
| tool_use (MCP) | `graph/builder.py` · `_sources` → `GovernedMcpToolSource` | The only tools are a remote server's, arriving as governed `ToolDefinition`s |
| guardrails | `graph/agent.py` · `_screened` | The person's message is screened for injection before the model reads it |
| hallucination_check | `graph/nodes.py` · level 1 `ConfidenceCheck`, then `guardrails/figures.py` · `FigureCheck` | Every figure must appear in a query result |
| human-verifiable output | `presentation/reply.py` | The reply carries the SQL it ran and the rows it rests on |

### Governance and LLD

| Pattern | Where |
|---|---|
| gatekeeper (authority only) | `graph/tool_calls.py` · `GovernedCalls` — VALID, VOID, or UNGOVERNED when withheld |
| adapter | `agent_platform/knowledge/mcp/client/` (SDK session), `agent_platform/runtime/executor/registry_handler.py` |
| active object | `agent_platform/knowledge/mcp/client/` · one task owns the MCP session |
| strategy | `agent_platform/analytics/query_guard.py` |
| facade | `agent_platform/analytics` · `SourceDatabase` |
| factory / composition root | `graph/builder.py` |

## Platform mechanisms the patterns run on

| Mechanism | Package | Used by |
|---|---|---|
| Circuit breaker, retry, timeout, bulkhead, idempotency, saga, hedge, pub-sub | `resilience` | every agent (model chain, Executor, web gateway, sagas) |
| Template method (`LlmJudge`) | `judge` | eval graders, the research audit |
| Decorator (`RecordedClient`, `SteppedClient`, `PiiBoundary`, `RateLimitedTransport`) | `llm`, `security`, `resilience` | every model call |
| Observer (`StageReporter`, `ModelCallObserver`) | `observability` | every card |
| Strategy (context strategies, stop rules, stagnation tiers) | `context`, `loop_control` | every agent with `context` |
| Facade (`MemoryFacade`, `SecuritySuite`) | `memory`, `security` | every agent |
| Factory (`create_*`, `build_*`) | throughout | composition roots |
