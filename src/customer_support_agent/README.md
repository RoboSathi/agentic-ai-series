# customer_support_agent — the running project

A customer-support refund agent: it answers order and refund questions,
issues refunds within its limit, queues larger ones for a person, and
refuses what the policy forbids. It **composes** `agent_platform` rather
than duplicating it; everything here is this business's vocabulary and
rules. Business rules live in code (`domain/refund_policy.py`), never only
in the prompt.

Run it with `make agent`; seed its data with `make dummy-data`; evaluate it
with `make eval` (see [evals](evals/README.md)).

## Modules

| Module | Owns |
|---|---|
| [basic_sanity](basic_sanity/README.md) | `OrderSanityCheck` — pre-LLM order checks |
| [config](config/README.md) | This agent's overlay and settings |
| [console](console/README.md) | `CustomerSupportHost`, `/queue` |
| [context](context/README.md) | (empty — context sources are platform components) |
| [loop_control](loop_control/README.md) | Tool-use loop and output quality gate |
| [domain](domain/README.md) | Orders, customers, identity, the refund policy |
| [dummy_data](dummy_data/README.md) | The demo shop and `make dummy-data` |
| [evals](evals/README.md) | Golden set, oracle, rubric, security suite |
| [gatekeeper](gatekeeper/README.md) | Business rules and mandates for the Gatekeeper |
| [graph](graph/README.md) | The LangGraph app and `build_agent` |
| [intent](intent/README.md) | Intent routing and follow-ups |
| [knowledge_base](knowledge_base/README.md) | Articles retrieved as semantic memory |
| [observability](observability/README.md) | Turn accounting for SLOs |
| [persistence](persistence/README.md) | Schema and repositories |
| [presentation](presentation/README.md) | Rendering orders and stage cards |
| [prompts](prompts/README.md) | System prompts and the policy text |
| [resilience](resilience/README.md) | `RefundSaga` |
| [review](review/README.md) | The human review queue |
| [security](security/README.md) | Customer authentication |
| [tools](tools/README.md) | The four `support.*` tools |
| [verification](verification/README.md) | Reading refunds back |
