# intent — what the customer is asking for

`CustomerSupportIntentRouter`: decides the intent of a message and which
tools it unlocks — deterministic rejects first, then the local classifier,
then phrase rules — and recognises follow-ups that continue the previous
subject ("what about 10010?").

## Vocabulary

- **intent** — `ORDER_STATUS`, `ORDER_DETAILS`, `REQUEST_REFUND`, … with a mode (`read`/`write`) and its tools
- **disposition** — supported, needs clarification, or out of scope
- **follow-up** — a message that restates nothing and continues the last subject

## Public interface

- `CustomerSupportIntentRouter(backend=...)`, `Intent` (`router.py`)
- `load_intent_classifier()` — warms the shared classifier before an event loop opens

## Configuration

`customer_support_agent.intents`, `default_intent`, `intent_router`, `followup`; the model in `llm.intent_classifier` (this agent's overlay).

## Seams

- A backend can be injected (tests use a fake).

## Collaborators

- `agent_platform.intent`.

## Episodes

Pre-LLM intent routing.

## Tests

Run with `./venv/bin/pytest tests/test_customer_support_intent.py tests/test_customer_support_intent_routing.py tests/test_intent_followups.py -q` (or `make test` for everything).
