# console — this agent in the operator console

`CustomerSupportHost`: builds the agent, warms the intent classifier and NLI before the event loop, syncs the knowledge base, recovers orphaned runs, settles issued refunds as housekeeping, and adds `/void`, `/cancel`, `/kb`, `/skills` and the support review handler.

- **Entry points:** `CustomerSupportHost`, `VoidCommand`, `SupportReviewHandler` (`review_handler.py`)
- **Configuration:** `agent.host`
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent)
