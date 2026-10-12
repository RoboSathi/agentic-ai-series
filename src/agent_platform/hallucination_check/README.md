# hallucination_check — claim-level grounding

Checks a draft reply claim by claim: level 1 token confidence, level 2 the agent's grounding (direct check → fact ledger → state model → NLI, in one consistent world). `evidence/` holds tool observations as premises and the append-only evidence ledger with the verbatim-span rule.

- **Entry points:** `DetectionLadder`, `GroundingResolver`, `ClaimClassifier`, `tool_evidence`, `EvidenceLedger`, `VerbatimSpanValidator`
- **Configuration:** `hallucination_check:`
- **Design:** [LLD §A17](../../../design_docs/LLD.md#a17-hallucination_check) · [ADR P-17](../../../design_docs/ADR.md#p-17--claims-are-grounded-in-one-consistent-world-cheapest-evidence-first)
