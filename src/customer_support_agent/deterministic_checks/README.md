# deterministic_checks — this agent's pre-flight checks

`OrderChecks`: when a message names an order, check it exists, is the authenticated customer's, is not already refunded and is refundable — before any model call. Implements the platform's `DeterministicChecker`.

- **Entry points:** `OrderChecks` (`order_validator.py`)
- **Configuration:** none (`features.deterministic_checks`)
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent) · [ADR P-08](../../../design_docs/ADR.md#p-08--deterministic_checks-is-the-seam-for-an-agents-business-logic)
