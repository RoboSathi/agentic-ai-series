# intent — what the customer is asking for

`CustomerSupportIntentRouter`: decides the intent of a message and which
tools it unlocks — deterministic rejects first, then phrase rules, then the
local classifier at its own threshold — and recognises follow-ups that continue
the previous subject ("what about 10010?").

## Vocabulary

- **intent** — `ORDER_STATUS`, `ORDER_DETAILS`, `REQUEST_REFUND`, … with a mode (`read`/`write`) and its tools
- **disposition** — supported, needs clarification, or out of scope
- **follow-up** — a message that restates nothing and continues the last subject
- **classifier** — the model behind a turn the rules did not settle: Laya (default), asked which
  `intents[].description` fits, or the fine-tuned BERT through its label mapping

## Public interface

- `CustomerSupportIntentRouter(backend=...)`, `Intent` (`router.py`)
- `load_intent_classifier()` — warms the shared classifier before an event loop opens
- `intent_choices()` — this agent's intents as a description-reading classifier is given them
- `rank_with_model(text)` — the classifier's view of one message alone, in this agent's intents

## Configuration

- `llm.intent_classifier.provider` — which classifier: `laya_local` (default) or `huggingface_local` (BERT).
- `intent_router.classifiers.<provider>` — that classifier's `confidence_threshold` (Laya 0.6,
  BERT 0.75) and `model_label_mapping`. Read once, when the router is built.
- `intent_router.phrase_rules` — route by `intents[].phrases` before the classifier (default on).
- `intents`, `default_intent`, `intent_router.instructions` (the question Laya is asked), `followup`.
- Profiles: `bert_intent` (BERT instead of Laya), `no_phrase_rules` (rules and regex rejects off).
- Comparison: `make eval csa STAGE=intent` (`evals.intent_bakeoff`) scores every contender.

## Seams

- A backend can be injected (tests use a fake).

## Collaborators

- `agent_platform.intent`.

## Episodes

Pre-LLM intent routing.

## Tests

Run with `./venv/bin/pytest tests/test_customer_support_intent.py tests/test_customer_support_intent_routing.py tests/test_intent_followups.py tests/unit/test_laya_intent_routing.py tests/unit/test_intent_bakeoff_runner.py -q` (or `make test` for everything).
