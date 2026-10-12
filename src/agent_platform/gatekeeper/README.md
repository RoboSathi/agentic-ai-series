# gatekeeper — deterministic authority

Decides whether the agent has the authority to run the exact call proposed — offered, mandate, RBAC, authority rank, the agent's own bounds — and names a standing (VOID, RATIFIABLE, VOIDABLE, VALID, UNGOVERNED). Calls no model and reads no database per call.

- **Entry points:** `Gatekeeper`, `GatekeeperImpl`, `DeterministicBusinessRules`, mandates (`mandate.py`), `CumulativeLedger`, `Standing`
- **Configuration:** `gatekeeper:`
- **Design:** [LLD §A10](../../../design_docs/LLD.md#a10-gatekeeper) · [ADR P-09](../../../design_docs/ADR.md#p-09--the-gatekeeper-decides-authority-only)
