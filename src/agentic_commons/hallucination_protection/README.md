# hallucination_protection — claim-level grounding

Checks a draft reply claim by claim against the evidence the agent
actually had: tool results, policy text, retrieved knowledge. Each claim is
retrieved against, checked with a local NLI model, and resolved by evidence
authority when evidence conflicts. Also diagnoses whether a failure was
grounding or reasoning, and tracks how well confidence predicts correctness.

## Vocabulary

- **atomic claim** — one checkable assertion split out of a reply
- **evidence tier** — how authoritative a source is (tool result > policy > system/history); support-only tiers can support but never refute
- **record claim** — a claim about the customer's own record — only tool results may support it
- **undetermined** — neither supported nor refuted — counted, not guessed

## Public interface

- `ClaimDecomposer`, `GroundingResolver`, `DetectionLadder` (`checker.py`)
- `GroundingReasoningDiagnoser`, `TrustCalibrationTracker`
- `AtomicClaim`, `EvidenceCheck`, `DetectionVerdict`, `GenerationAssessment`, `TrustReport` (`types.py`)

## Configuration

Section `hallucination_protection:` — thresholds (`nli_entailment_threshold`, `confidence_threshold`,
`evidence_min_similarity`, `embedding_top_k`), `evidence_authority`, `support_only_tiers`,
`record_claim_pattern` / `record_claim_tiers`, `combined_premise_enabled`, self-consistency settings,
`max_repair_attempts`. Built only when `features.hallucination_guard` is on.

## Seams

- The ranker and entailment model are injected; NLI runs locally (`llm.local_nli`).

## Collaborators

- Used by `guardrails.ResponseGuardrail`; its per-turn rate is compared with the eval judge's verdicts in the judge report (κ).

## Episodes

E25, E26, E27.

## Tests

Run with `./venv/bin/pytest tests/unit/test_hallucination_protection.py tests/test_hallucination_guardrails.py -q` (or `make test` for everything).
