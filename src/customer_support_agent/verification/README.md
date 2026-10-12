# verification — reading back what a refund did

`RefundEffectSource` reads the refunds ledger and the order back so the platform's `ReadBackVerifier` confirms the money moved before any reply claims it did.

- **Entry points:** `RefundEffectSource` (`refund_effects.py`)
- **Configuration:** none
- **Design:** [ADR P-18](../../../design_docs/ADR.md#p-18--verified-effects-not-executor-success-are-evidence)
