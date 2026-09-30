# ADR ra-0001 — Hybrid interview termination, enforced in code

## Context

An interviewer that decides for itself when it has asked enough will stop early whenever the user sounds impatient; one that never stops is useless. Prompt instructions ("don't finish until…") are advice to the model, not a rule.

## Decision

Termination is **hybrid and enforced in code** (`scoping/termination.py`). The interviewer may only *propose* done — and `TerminationPolicy` lets the proposal through only when no required checklist item is `open`; the user confirms. The user may say `/done` at any time; every open item is written to the Brief as a **known unknown**, which the rubric must acknowledge (an "acknowledged and bounded" item per unknown) and the report must list. At `scoping.max_turns` the agent stops asking, lists what is open and offers `/waive <id|all>` or `/done`.

## Consequences

- The user always has the last word; the model never does.
- Closing early is honest rather than silent: `ds-closed-early` and `ex-iron-air-waived` exercise it; `ds-closed-before-criteria` shows that closing before anything is scorable yields `PARTIAL`, not an empty `COMPLETE`.
- The turn cap bounds cost even with an uncooperative user.

## Alternatives rejected

- **Model decides** — sycophantic under pressure, untestable.
- **Fixed question list** — cannot follow up on vague answers.
- **User decides only** — the agent cannot say "I have what I need".
