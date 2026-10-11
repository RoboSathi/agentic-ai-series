# ADR ra-0022: Reflection on the report, an independent critic, and lessons across runs

**Status:** accepted 2026-10-11 · amends ra-0002 (rubric text and the cross-check)

## Context

The run's critics found faults but never handed them back to the writer:

- `reflect` critiqued the evidence and planned more research (Producer-Critic over gathering).
- The citation guard cut unsupported sentences.
- The audit scored the report.

Every must-pass failure went to gap-fill, which searched again. The report was then drafted
from scratch by a synthesizer that never saw its earlier draft or the audit's reasons. The
judge ran on the writer's own model (`independent_judge: false`), so the writer graded
itself. No run learned from another.

The rubric also over-claimed. "The sub-concept 'X' is explained and cross-checked in two
sources" was measured as two hosts whose claims mention X: no entailment and no agreement.
A vague interview answer could become a subject (`Concepts: all` → "the sub-concept 'all'").

## Decision

1. **Route audit failures by fault.**
   - A failure of a check in `reflection.writing_checks` is a writing fault, sent to `revise`. These checks are the judge's, the cited-sentence entailment, citations, links, counter-evidence and known unknowns.
   - Any other failure is an evidence gap, sent to gap-fill.
   - Evidence goes first: a report revised before its evidence is in would be revised again.
2. **Producer-Critic on the report.** `revise` hands the writer its previous sections and the audit's findings: each criterion, the judge's reason and the sentences that drove the score. The revision then goes through the same citation guard, self-check and audit as the draft.
   - It is bounded by `max_revisions`.
   - It stops on a plateau, when the score gain is below `min_score_gain`.
   - After a gap-fill, the re-draft also receives the previous report and the findings.
   - `revise` is a node only when reflection is declared.
3. **An independent critic.** The judge's chain excludes the writer's primary model.
   - At run time, an audit in which any criterion was judged by the model that actually wrote the report (a backup counts as itself) is recorded as not independent.
4. **Reflexion.** At archive, `LessonBook` writes the run's lessons to episodic memory. The lessons are written in code from verdicts, never by a model, and cover:
   - failed criteria
   - the most frequent rejection codes
   - budget limits hit
   - a revision's score gain

   At the next run's plan, lessons from earlier runs of the same kind whose topics share words with this one are recalled. They are shown on a LESSONS card and given to the planner, the reflector and the reviser, which are told to use them to avoid repeating a failure and never as evidence. The `reflexion` feature needs `memory.episodic`.
5. **Honest cross-check.** `claim_verdict` and `concept_covered` are now NLI checks.
   - They count only the domains whose claims about the subject are entailed by their own span, and that no other domain's claim contradicts.
   - At most `max_pairs` pairs are compared.
   - The rubric text says exactly that.
6. **No vague subjects.** An answer to a subject item made only of `vague_subject_words` leaves the item open, to be asked again with examples. After `vague_answers_before_delegating` such answers the item is `delegated`: the interviewer proposes concrete values, which are marked `derived` and shown at rubric review.

## Consequences

- A run's reflection now covers the whole cycle: the evidence (reflect, gap-fill), the report (audit → revise) and earlier runs (lessons).
- The judge runs on lightning, then gpt-oss, both weaker than the writer. A criterion they cannot score is "not judged", never self-graded.
- With no round left, the critique is no longer asked: its follow-ups could only end in "no room".
