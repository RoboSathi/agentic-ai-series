# runtime/verification — did the side effect happen?

After a side-effecting call, reads back what should have changed through an agent's `EffectSource`; an unverified result never becomes authoritative evidence. Also the named acceptance-check registry.

- **Entry points:** `ReadBackVerifier`, `EffectSource`, `IntendedEffect`, `VerificationReport`, `AcceptanceCheckRegistry` (`acceptance.py`)
- **Configuration:** none
- **Design:** [LLD §A9](../../../../design_docs/LLD.md#a9-runtime-executor-verification-scheduling) · [ADR P-18](../../../../design_docs/ADR.md#p-18--verified-effects-not-executor-success-are-evidence)
