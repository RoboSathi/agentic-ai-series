# Customer support agent — patterns

Where this agent uses each design pattern, derived from the code as it stands.
Episodes demonstrate a pattern by running this agent, not a separate reference copy.

## Workflow and agentic patterns

| Pattern | Where | Notes |
|---|---|---|
| sequence | graph/edges.py · the front-door stages (pii_filtering → prompt_injection → content_moderation → classify_intent → assemble_context) | fixed order; a withheld stage is not built and its neighbours are wired together |
| branch | graph/edges.py · _after_authenticate, _after_screening, _after_finalize | deterministic predicates halt the turn or continue it |
| router | intent/router.py · classify_intent → read_intent | deterministic out-of-scope rejection, then a small classifier, then one clarification question |
| tool_use | graph/nodes/propose.py ⇄ graph/nodes/execute.py | the model proposes a typed tool call, execute runs it through the Gatekeeper, bounded by max_tool_iterations |
| human_in_the_loop | graph/nodes/review_wait.py · await_approval (interrupt) + review/queue.py | above-limit refunds pause the durable run until a reviewer decides |
| rag_pipeline | graph/builder.py · FiveStagePipeline with KnowledgeBaseContextSource (assemble_context) |  |
| memory_systems | graph/builder.py · MemoryFacade over the four CoALA stores; admin/dream.py consolidation | procedural: `prompts/skills/*/SKILL.md` synced into `ProceduralStore`, one per intent, selected by `intent:`/`route:` signals (ADR 0010); dream lessons enter via `/skills approve` |
| profile | prompts/AGENT.md · `build_system_prompt` | the agent's stable operating profile, always in the prompt; procedures follow it per turn |
| planning | loop_control/quality_gate.py · PlanLookup | a config lookup of the tool order per intent, committed before acting |
| reflection | loop_control/quality_gate.py · GroundedCritic | one deterministic critique of the draft against tool outputs |
| evaluator_optimizer | loop_control/quality_gate.py · OutputQualityGate (critique, at most one repair, IndependentScorer) |  |

## LLD and resilience patterns

| Pattern | Where |
|---|---|
| saga | resilience/refund_saga.py · resilience/cancellation_saga.py |
| retry + idempotency + circuit breaker + timeout | graph/nodes/propose.py · the model call through ResilientCaller |
| strategy | loop_control/quality_gate.py · Planner / Reflector / Evaluator protocols |
| factory | graph/builder.py (composition root) |
| facade | agent_platform MemoryFacade |
