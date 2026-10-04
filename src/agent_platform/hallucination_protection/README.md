# hallucination_protection — claim-level grounding

Checks a draft reply claim by claim against the evidence the agent actually
had. Each claim is first **classed**, and the class decides which tiers may
ground it — so a courtesy is never scored, a promise is grounded by the rule
that scripts it, and a fact about the case is grounded only by a record.
Also diagnoses whether a failure was grounding or reasoning, and tracks how
well confidence predicts correctness.

## Vocabulary

- **atomic claim** — one checkable assertion split out of a reply
- **claim class** — `record` (a fact about this case: an order, an amount, an outcome stated as done), `rule` (how things work, and every promise: "you'll hear back soon"), `conversational` ("Thanks for reaching out" — not a claim, never scored)
- **tier** — one stage of the cascade: which evidence kinds it reads, which claim classes it may ground, whether it may refute
- **direct check** — the model-free step for a claim about the case: an id or amount the record lacks, the refund state machine, and the record of a refused call. It refutes, and it supports what it can place
- **state model** (`states.py`) — the refund state machine (`none → pending_review → approved → completed`, or `denied`). A claim about where the refund stands is judged against the *latest* recorded status for the order it names; "submitted" is true once the refund has got there, "pending review" only while it is there. A state record is read here and never sent to NLI, which took `pending_review` to contradict "submitted"
- **refusal** — a record claim that the agent did not act ("I'm unable to process a refund for order N"); supported by the `NOT EXECUTED` entry for that order, not by NLI
- **rule phrase** — a phrase carved out of an outcome and checked as the rule it states ("… to your original payment method")
- **undetermined** — nothing decided it; counted and shown, never blocking

## How a claim is grounded

```
record claim  →  direct check  →  tool_result
rule claim    →  direct check* →  policy → system prompt → memory
conversation  →  not checked
```

\* a date, weekday or number the claim brings that no evidence contains.

The deterministic system-of-record tier is one of **admission**, not of
prose: `EffectVerifier` re-reads the record after a write, and a call whose
effect did not take reaches grounding as `NOT EXECUTED` rather than as a
result to believe, so the direct check refutes any claim that it happened.
BasicSanity is a pre-flight gate on the state *before* the action; it halts a
bad turn and is not evidence, because it grounds no outcome and is stale by
reply time.

A read the turn has since overtaken is taken out first: an observation whose tool is
side-effecting (`state_changing`) makes every *earlier* read that names the same order stale, so
"No refund has been requested" cannot contradict a reply written after the refund was referred.
A later *re-read* through the same tool (same `source`) overtakes an earlier read of the same
subject too, so a settlement sweep between two reads leaves only the second. Subjects are the
agent's `refute_identifier_pattern` matches, not bare numbers, so a shared year or amount is not
a shared order. A refused call observed nothing and overtakes nothing; a source-less read is no
tool's re-read. A read about another order, or in a turn that neither wrote nor re-read, stands.

What stands is one consistent world, so within a tier **any entailment outvotes any
contradiction**: the evidence cannot both entail and contradict a claim, and when NLI says it
does the contradiction is about another attribute ("the refund for order 10842 has been
approved" against "order 10842 was cancelled"). A contradiction stands only when nothing current
in the tier entails the claim. Neither call order nor closeness decides — "the latest
observation wins" let a later refund read overrule an earlier order read it said nothing about.
Support, like refutation, must be about what the claim names: evidence naming only other
identifiers cannot support. Refund-state claims meet the deterministic state model before any
NLI. When the state model or the record of a refused call decides a claim,
the card says what was read; the other words in the sentence are not checked.

A refutation from any tier that may refute stands; otherwise the first tier
to support gets the credit; otherwise the claim is undetermined, and its rationale says, tier by tier, why. The system
prompt and memory support only. The customer's own words are in no tier. A
fact about the case with no record at all is refuted as unsourced.

Within a tier the surest result decides, support first (above). A rule that restates the claim in nearly its
own words (`rule_restatement_coverage`) supports it directly: NLI on an
*instruction* against the statement it dictates is not reliable.

## Public interface

- `ClaimClassifier`, `ClaimDecomposer`, `split_sentences`, `sentence_pieces`, `plain_text` (`claims.py`): a line break ends a claim, markdown is stripped, a lead-in line ("Tomorrow in Pune:") claims nothing; `StateModel` (`states.py`)
- `tool_evidence`, `Observation` (`evidence.py`): tool results as one `tool_result` premise per sentence, in call order — every agent builds its grounding evidence here
- `GroundingResolver` (`resolver.py`), `DirectChecker` (`direct.py`)
- `DetectionLadder`, `GroundingReasoningDiagnoser`, `TrustCalibrationTracker` (`checker.py`)
- `AtomicClaim`, `ClaimClass`, `EvidenceCheck`, `DetectionVerdict`, `GenerationAssessment`, `TrustReport` (`types.py`)

## Configuration

Section `hallucination_protection:` — thresholds (`nli_entailment_threshold`,
`evidence_min_similarity`, `embedding_top_k`, `rule_restatement_coverage`), the
`tiers` list (order, kinds, claim classes, `may_refute`), the classifier vocabulary (`conversational_openers`,
`record_identifier_pattern`, `outcome_pattern`, `promissory_pattern`), the
the refund state machine (`refund_states`: `ladder`, `branches`, `observed_as`, `phrases`; plus `refund_state_pattern`, `refusal_pattern`), `rule_phrases`, self-consistency
settings, `max_repair_attempts`. Built only when `features.hallucination_guard`
is on.

## Seams

- The ranker and entailment model are injected; NLI runs locally (`llm.local_nli`).
- Evidence arrives as `Chunk`s whose `kind` names the tier; the agent decides what it puts in (verified tool results, policy, the prompts a reply was written under as `system`, verified memory).
- What a removed sentence is: claims carry the index of the sentence they came from, and a refuted claim removes that sentence, never a text match.

## Collaborators

- Used by `guardrails.ResponseGuardrail`; its per-turn rate is compared with the eval judge's verdicts in the judge report (κ).

## Episodes

E25, E26, E27.

## Tests

Run with `pytest tests/unit/test_grounding_cascade.py tests/unit/test_grounding_evidence.py tests/unit/test_hallucination_protection.py tests/test_hallucination_guardrails.py -q` (or `make test` for everything).
