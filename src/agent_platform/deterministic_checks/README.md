# deterministic_checks — the business-logic seam

The contract for checks that run before any model is asked: a request in (with an opaque `subject_id`), findings out. Every concrete checker belongs to an agent.

- **Entry points:** `DeterministicChecker`, `DeterministicCheckRequest`, `DeterministicCheckFinding`, `DeterministicCheckResult`
- **Configuration:** none (`features.deterministic_checks`)
- **Design:** [LLD §A13](../../../design_docs/LLD.md#a13-deterministic_checks) · [ADR P-08](../../../design_docs/ADR.md#p-08--deterministic_checks-is-the-seam-for-an-agents-business-logic)
