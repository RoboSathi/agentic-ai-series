# scoring — the NOLLM scoring cascade (E10)

Runtime infrastructure, not an eval: retrieval relevance, tool fitness, grounding and memory
consolidation all resolve claims here, cheapest tier first — exact assertion → deterministic
rule → embedding similarity → NLI entailment → calibrated LLM judge → human adjudication. A
tier that cannot decide declines so the next may try; reaching the judge is a failure to
resolve cheaply, not a first move. Moved out of `evals/` (2026-10-12): nothing here grades an
agent's suite.

A leaf package: it imports configuration, embeddings and logging, nothing that imports it
(`hallucination_check` → `context` → `tools` → `guardrails` → `hallucination_check` is an
existing cycle this package stays out of).

## Public interface

- `create_scoring_cascade()`, `NollmScoringCascadeImpl` (`nollm_scorer_ratio`, `tier_counts`)
- Scorers: `ExactAssertionScorer`, `DeterministicRuleScorer`, `EmbeddingSimilarityScorer`,
  `NliEntailmentScorer`, `CalibratedJudgeScorer` (needs a `CalibratedJudge`), `HumanAdjudicationScorer`
- Entailment: `EntailmentModel`, `LexicalEntailmentModel`, `create_entailment_model`,
  `preload_entailment_model`
- Types: `Claim`, `Verdict`, `ScorerTier`

## Configuration

`hallucination_check.scoring` — `tiers` (per-tier toggles), the embedding and NLI thresholds,
`judge_confidence_threshold`, and the lexical fallback (`nli.min_overlap`, `negation_markers`).

## Tests

`./venv/bin/pytest tests/unit/test_scoring_cascade.py -q`
