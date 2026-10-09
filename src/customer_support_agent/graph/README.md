# graph — the LangGraph application

The agent itself: its state, nodes, edges, and the composition root.
Order: authenticate → choose order → screening (basic sanity, PII,
prompt injection, content moderation — only those declared) → intent →
context → propose ⇄ execute → finalize → await approval. A deterministic read
route answers straight from the database with no model call.

## Vocabulary

- **composition root** — `build_agent` — reads the feature tree once and builds only what is declared
- **AgentState** — what flows between nodes: query, redacted query, customer, working memory, decisions, response, metadata
- **durable turn** — every turn runs as its own checkpointed thread

## Public interface

- `build_agent(llm=, pool=, gatekeeper=, working_memory=, cascade=, audit=, reporter=, with_memory=, observe=, basic_sanity=, intent_router=, features=, knowledge_base=, checkpointer=, episodic_store=)` (`builder.py`) — anything injected is used as is
- `CustomerSupportAgent.handle(query, session_id, customer)` → `AgentState`; `resume(thread_id, decision)` (`agent.py`)
- `build_review_service()` — the operator side of HITL
- Nodes (`nodes/`): authenticate, screening, intent, context, reason, act, finalize, review_wait

## Configuration

No section of its own; the builder reads every section it wires.

## Seams

- Every collaborator is injectable — tests, episodes and the eval harness each substitute what they need.

## Collaborators

- Composes nearly every platform module; the platform never imports it.

## Episodes

Every converted episode runs this graph (E13, E22, E30, E35, …).

## Tests

Run with `./venv/bin/pytest tests/unit/test_customer_support_agent.py tests/integration/test_request_lifecycle.py -q` (or `make test` for everything).
