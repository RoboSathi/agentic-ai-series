# ADR 0011 — CSA classifies intent with Laya; the classifier is a config switch

- Status: accepted; implemented 2026-10-10
- Date: 2026-10-10

## Context

The support agent routes each turn to one of 7 intents. Phrase rules settle the obvious
turns, and a local classifier decides the rest. That classifier was a fine-tuned BERT
(`learn-abc/magicSupport-intent-classifier`) with 28 generic labels, folded into our 7
by `model_label_mapping`.
- ORDER_DETAILS and LIST_ORDERS had no label at all, so the model could never choose them.
- 21 of the 28 labels fell into UNKNOWN.
- Its softmax sat near 1.0 even when it was wrong, so the routing threshold could not
  separate a safe decision from an unsafe one.

Laya (`convaiinnovations/laya`, `typed-decisions` checkpoint, Apache-2.0) is a ModernBERT
decision model. It answers one choice question whose options are our intents'
descriptions, and returns a probability for each option.

## Evidence

Bake-off on 50 labelled messages (`evals/intent/cases.yaml`, `make intent-bakeoff csa`),
each classifier at its own threshold:

| | Model alone | Router, rules on | Router, rules off | ECE |
|---|---|---|---|---|
| BERT @0.75 | 28% | 38% (15 routed, 12 right) | 28% (10 routed, 7 right) | 0.65 |
| Laya @0.6 | 72% | **56% (22 routed, 21 right)** | 50% (19 routed, 18 right) | 0.21 |

- Laya reaches every intent.
- At 0.6 it is right 95% of the time it decides; the rest of the turns go to the LLM.
- It sends fewer read questions to a write intent: 1, against BERT's 3.
- It costs about 215 ms per message on CPU, against BERT's 30 ms.
- Phrase rules still add +3 messages for Laya, mostly answers it got right below 0.6.

Limits: n=50, so each figure carries about ±13 points. The labels were drafted for the
test. Single-turn messages only.

## Decision

- **Laya is CSA's default intent classifier** (`llm.intent_classifier.provider: laya_local`).
- **The classifier is one config switch.** Each provider's threshold and label mapping
  sit with it in `intent_router.classifiers.<provider>`: Laya at 0.6 with identity labels,
  BERT at 0.75 with its mapping. The router reads them once, when it is built.
  `profiles/bert_intent.yaml` restores BERT.
- **Phrase rules stay on, as their own switch** (`intent_router.phrase_rules`).
  `profiles/no_phrase_rules.yaml` turns them off, together with the regex rejects. The
  order validator reads `intents[].phrases` whatever the switch says.
- **The platform stays agent-agnostic.** `LayaIntentClassifier` takes its labels and
  question from the agent at build time (`create_intent_classifier(choices, instructions)`).
- `laya` becomes a core dependency. A checkpoint is loaded once per process.
  `make model-check` probes whichever classifier is selected.

## Consequences

- Turns the rules miss cost about 185 ms more on CPU. In exchange, the router decides 7 more
  of the 50 turns itself rather than sending them to the LLM, so fewer turns pay for an LLM call.
- An intent is added by writing its description; nothing is retrained.
- Open items:
  - review the 50 labels;
  - fix the REFUND_STATUS phrase gap ("when will i get refund…" matches REQUEST_REFUND's
    "refund for order");
  - refit Laya's temperature on real traffic before trusting 0.6 as a probability.
