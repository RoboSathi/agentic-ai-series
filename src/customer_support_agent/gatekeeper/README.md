# gatekeeper — this agent's rules for the platform Gatekeeper

Two authority rules, neither about the order's own correctness: a refund or cancellation above the per-action limit (the persisted policy row) escalates to a person (RATIFIABLE), and a write may only target the order the conversation established (else VOID).

- **Entry points:** `CustomerSupportBusinessRules` (`business_rules.py`), `EstablishedOrderScope` (`mandates.py`)
- **Configuration:** `gatekeeper.mandate` in this agent's overlay
- **Design:** [ADR P-09](../../../design_docs/ADR.md#p-09--the-gatekeeper-decides-authority-only) · [ADR C-02](../../../design_docs/ADR.md#c-02--the-refund-limit-lives-in-the-database-and-is-enforced-by-the-gatekeeper)
