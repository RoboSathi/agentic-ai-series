# ADR ra-0020 — Two paths: a plain question is answered, `/research` researches

**Status:** accepted 2026-10-04 · supersedes ra-0017 · amends ra-0011

## Context

ra-0017 gave the agent two *modes*, Q&A and Research, chosen by a gate that read every first
message; the direct answer (a ReAct loop over read-only tools) later took over Q&A, and an
answer-or-research intent router was put in front of it. The result was three classifiers
deciding overlapping things: the router guessed whether a plain message wanted research, and
the scope gate — reached only through `/research`, which had already settled that — still
weighed "qa or research" and offered a reframe menu. Its card read "explainer … a direct
question suitable for a QA answer" for a question the user had explicitly asked to research.

## Decision

Two paths, chosen by the user, never guessed:

- **A plain message** is answered directly: the ReAct loop with web tools, grounded against
  what the tools returned. No scope, no interview.
- **`/research [question]`** starts a research scope. Its first step is the **research kind**
  (`decision_support`, `landscape_brief`, `fact_verification`, `explainer`, or `general`),
  which picks the interview checklist, the rubric template, the stop rule and the report's
  sections. Cue rules, then the example classifier (nearest labelled examples over a local
  embedder), then the LLM decide it.

Removed: the intent router, `/qa`, the Q&A mode and archetype, `auto_freeze`, the reframe menu
and the `research_cues`. The refusal patterns stay, applied on both paths before any model.

## Consequences

- A research-shaped plain question is answered directly; the answer may say that `/research`
  runs a full, audited report. Nothing starts an interview the user did not ask for.
- The only classifier left decides *which kind* of research, which changes what the run
  checks and when it stops — so it improves results, not speed.
- With `direct_answer` withheld, a plain message starts a research scope (no answerer exists).
