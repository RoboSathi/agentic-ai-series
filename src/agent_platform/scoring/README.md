# scoring — the NOLLM scoring cascade

Claims resolved cheapest tier first: exact assertion → deterministic rule → embedding similarity → NLI → calibrated judge → human. Runtime infrastructure, and a leaf package.

- **Entry points:** `create_scoring_cascade`, `NollmScoringCascadeImpl`, one scorer per tier, `EntailmentModel`, `create_entailment_model`
- **Configuration:** `hallucination_check.scoring`
- **Design:** [LLD §A6](../../../design_docs/LLD.md#a6-scoring)
