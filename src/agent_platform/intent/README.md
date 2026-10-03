# intent — reusable intent classification

Contracts and a local Hugging Face provider for classifying what a
message is asking for. Which intents exist, their tools and thresholds are
an agent's business; this module only runs a classifier and reports
candidates with scores.

## Vocabulary

- **IntentCandidate** — one label with its score
- **IntentAssessment** — the chosen intent, disposition (`supported`, `clarify`, `out_of_scope`), candidates and source
- **disposition** — what the router decided to do with the classification

## Public interface

- `IntentClassifierInterface` (Protocol), `HuggingFaceIntentClassifier`, `create_intent_classifier` (`classifier.py`)
- `IntentAssessment`, `IntentCandidate`, `IntentDisposition` (`types.py`)

## Configuration

`llm.intent_classifier:` — `enabled` (use the model; off → phrase rules only), `provider`, `model`,
`local_model_dir`, `download_if_missing`, `max_length`. The agent overlay names its model. Whether the
stage exists: `features.intent_classification`.

## Seams

- An agent's router wraps a classifier. Load it before an event loop opens (macOS tokenizer start-up).

## Collaborators

- Uses `llm.local_models`; consumed by the agent's intent node.

## Episodes

Pre-LLM intent routing (see the agent's intent module).

## Tests

Run with `./venv/bin/pytest tests/test_customer_support_intent.py tests/test_customer_support_intent_routing.py -q` (or `make test` for everything).
