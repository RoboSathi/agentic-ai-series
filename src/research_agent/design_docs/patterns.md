# Research agent — patterns

Where this agent uses each design pattern, derived from the code as it stands.
Episodes demonstrate a pattern by running this agent, not a separate reference copy.

## Workflow and agentic patterns

| Pattern | Where | Notes |
|---|---|---|
| router | scoping/kind.py · research_kind node | a /research question is routed to its research kind (cue rules, then the example classifier, then the LLM) or refused; the kind picks the checklist, rubric, stop rule and report |
| human_in_the_loop | graph/scoping_graph.py · await_user (durability/hitl.await_decision) | the interview and rubric review pause the durable graph until the user answers; /approve freezes the contract |
| planning | planning/planner.py + planning/dag_validation.py · plan_dag (with its validation) | the planner decomposes the contract into a DAG of sub-questions, validated (acyclic, bounded, traced) before anything runs |
| fan_out_fan_in | graph/run_graph.py · batch_start → research_node (Send per ready node) → batch_done | each batch's ready nodes run concurrently under the scheduler's cap and join at the batch boundary |
| react | research/worker.py · research_node | a bounded act–observe loop per sub-question (search, fetch, read, follow up); next actions come from the plan and reflection, not free-form thoughts |
| agentic_retrieval | research/worker.py + research/reflector.py | the agent chooses what to search next from what it has found (reflection's follow-up queries), within the budget |
| map_reduce | research/extractor.py (map) · synthesis/synthesizer.py (reduce) | each fetched page maps to verbatim claims in the ledger; synthesis reduces the ledger to the report |
| reflection (Producer-Critic) | research/reflector.py · reflect_gap_analysis; graph/run_nodes.py · independent_audit → revise | over the evidence: after each batch, gaps against the rubric become follow-up nodes or a stop decision. Over the report: the audit's writing findings (judge reasons, unsupported sentences) go back to the writer in a bounded revision, re-audited each time and stopped on a plateau (ra-0022). The critics are external checks (citations, NLI, a judge on another model), not the writer's opinion of itself |
| reflexion | reflexion/lessons.py · archive → plan | what a run's critics found is written to episodic memory as lessons (in code, from verdicts) and recalled for the next run of the same kind on a similar topic: planner, reflector and reviser are told to avoid repeating it |
| evaluator_optimizer | graph/run_nodes.py · synthesize → independent_audit → gap_fill / revise | an independent audit scores the report against the frozen rubric; evidence gaps drive one bounded gap-fill round, writing faults a bounded revision |
| orchestrator_subagent | audit/independent.py · IndependentAuditor (agent_platform.durability.IsolatedGraphRunner) | the independent audit runs as an isolated sub-graph that receives only (contract, report, ledger_snapshot) |
| memory_systems | archive/store.py · archive node | each run leaves an episodic index row and a semantic fact; working memory carries session status; procedural holds authored role skills only (`prompts/skills/`, ra-0021), prompted below the ROLE line with the profile |

## LLD and resilience patterns

| Pattern | Where |
|---|---|
| strategy | research/stop_rules/ (one stop rule per archetype) |
| adapter | agent_platform/web/providers.py · TavilyProvider and DuckDuckGoProvider behind SearchProvider |
| chain of responsibility | agent_platform/web/router.py · SearchRouter (free provider first, metered fallback) |
| factory | graph/builder.py (composition root) |
| builder | scoping/brief.py · BriefBuilder |
| observer | observability/telemetry.py (StageReporter + AuditLedger) · WebGateway observer |
| decorator | research_agent/evals/agent_under_test.py · RecordingLLMClient around every role client |
| facade | agent_platform MemoryFacade, used by archive/store.py |
| retry + idempotency + circuit breaker | agent_platform/web/gateway.py · live search and fetch through ResilientCaller |
| bulkhead | agent_platform/scheduling/scheduler.py · per-batch concurrency cap |
