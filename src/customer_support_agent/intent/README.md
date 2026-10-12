# intent — what the customer is asking for

`CustomerSupportIntentRouter`: deterministic rejects, phrase rules, then the local classifier (Laya by default, BERT by profile) at its own threshold; follow-ups; the read-intent suggestion for an unplaced turn.

- **Entry points:** `CustomerSupportIntentRouter`, `Intent`, `load_intent_classifier`, `intent_choices`
- **Configuration:** `customer_support_agent.intents`, `intent_router`, `llm.intent_classifier`
- **Design:** [ADR C-06](../../../design_docs/ADR.md#c-06--intent-laya-by-default-the-classifier-a-config-switch)
