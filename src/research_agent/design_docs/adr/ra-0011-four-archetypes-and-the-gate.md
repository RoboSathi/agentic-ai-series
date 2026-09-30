# ADR ra-0011 — Four archetypes and the scopability gate

## Context

"Research X" is not a task; it has no stopping point. Each kind of research needs its own questions, acceptance criteria, report shape and definition of done.

## Decision

Four archetypes — `decision_support`, `landscape_brief`, `fact_verification`, `explainer` — each with a coverage checklist, a rubric template, a report template and a **stop-rule strategy class** (one file each in `research/stop_rules/`). The **scopability gate** classifies the first message behind the `ArchetypeClassifier` Protocol: `scopable(archetype)`, `needs_reframe` (open-ended requests become the first interview turn) or `refuse`. Without a decision/purpose/deliverable the agent keeps interviewing; it never proceeds to research.

## Consequences

- Every archetype is exercised by at least three golden cases.
- The approved recency window is scope: hits outside it are never read (`lb-recent-players`).

## Alternatives rejected

- **One generic flow** — no meaningful stop rule.
- **Free-form model classification only** — the Protocol lets a small model replace it later.
