# intent — reusable intent classification

Contracts and two local providers for classifying what a message is
asking for: a fine-tuned Hugging Face sequence classifier, and a Laya
decision model asked one choice question over the agent's labels. Which intents exist, their tools and thresholds are
an agent's business; this module only runs a classifier and reports
candidates with scores.

## Vocabulary

- **IntentCandidate** — one label with its score
- **IntentAssessment** — the chosen intent, disposition (`supported`, `clarify`, `out_of_scope`), candidates and source
- **disposition** — what the router decided to do with the classification

## Public interface

- `IntentClassifierInterface` (Protocol), `HuggingFaceIntentClassifier`, `create_intent_classifier(choices, instructions)` (`classifier.py`)
- `LayaIntentClassifier` (`laya.py`) — labels and their descriptions are injected by the agent; the checkpoint is loaded once per process
- `IntentAssessment`, `IntentCandidate`, `IntentDisposition` (`types.py`)

## Configuration

`llm.intent_classifier:` — `enabled` (use the model; off → phrase rules only), `provider`
(`huggingface_local` | `laya_local`), `model`, `local_model_dir`, `download_if_missing`, `max_length`,
and `laya:` (`model`, `subfolder`, `local_model_dir`, `device`, `max_len`, `question_id`, `state_key`).
The agent overlay names its models; `laya_local` needs the agent's labels at build time. Whether the
stage exists: `features.intent_classification`.

## Seams

- An agent's router wraps a classifier. Load it before an event loop opens (macOS tokenizer start-up).

## Collaborators

- Uses `llm.local_models` (BERT) and the `laya` package (Laya); consumed by the agent's intent node.
- `make model-check` probes whichever provider is selected (`llm.check.local.intent_question/choices` for Laya).

## Episodes

Pre-LLM intent routing (see the agent's intent module).

## Tests

Run with `./venv/bin/pytest tests/agent_platform/test_laya_intent_classifier.py tests/test_customer_support_intent_routing.py -q` (or `make test` for everything).
