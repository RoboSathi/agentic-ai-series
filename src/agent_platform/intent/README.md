# intent — intent classification contracts and providers

Runs a classifier and reports candidates with scores: a fine-tuned sequence classifier, or a Laya decision model asked one choice question over labels the agent supplies. Which intents exist is the agent's business.

- **Entry points:** `IntentClassifierInterface`, `HuggingFaceIntentClassifier`, `LayaIntentClassifier`, `create_intent_classifier`, `IntentAssessment`
- **Configuration:** `llm.intent_classifier:`
- **Design:** [LLD §A14](../../../design_docs/LLD.md#a14-intent)
