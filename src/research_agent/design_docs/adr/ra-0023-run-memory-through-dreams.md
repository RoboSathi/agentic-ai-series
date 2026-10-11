# ra-0023 — What a run learned reaches the next run through dreams and skills

- Status: accepted; implemented 2026-10-11
- Supersedes: the Reflexion part of ra-0022. Builds on ADR 0012 and amends ra-0021.

## Context

- ra-0022 wrote "lessons" straight to episodic memory when a run was archived.
- At the next run's plan they were recalled by topic-word overlap and shown on a LESSONS card
  at the top of the run.
- Nothing consolidated them. A one-off failure and a recurring one looked the same.
- The card was noise, and the lessons bypassed the platform's dream and its governance.

## Decision

- **A finished run writes one dream event** (`memory/run_events.py`), kind `run`. Its signals
  are what the critics found:
  - `criterion_failed`: the rubric item's text
  - `source_rejected`: the rejection code
  - `budget_stop`: what ran out
- Every signal is scoped `archetype:<kind>`. Every turn also writes a turn event.
- **`make dream ra` reads them through `research_memory_profile()`**:
  - `RunSummary`: an episodic summary from the run's own record (kind, topic, status, the
    audit's scores, what failed).
  - `SourceReliability`: rejected domains become semantic facts, as system-of-record
    observations.
  - The platform's recurring-signal learner proposes a skill once a finding recurs in
    `skill_threshold` runs of one kind. The wording is in `memory.dream.learned_skill.lessons`.
- **A curator approves it in `/queue`**, with a comment. Until then it changes nothing.
- **At plan**, learned skills are selected without a model, by the `archetype:<kind>` signal and
  the topic. They are shown on the SKILLS card and handed to the planner, reflector and reviser
  as `skills_learned_from_past_runs`.
- `memory.procedural` is on again, for learned skills only. ra-0021 stands: no run role has an
  authored skill.
- The LESSONS card, `reflexion/`, the `reflexion` feature and its config are deleted.

## Consequences

- A lesson needs a pattern: one bad run no longer changes the next one's prompt.
- Each learned skill is a versioned, audited SKILL.md that a person approved and can retire.
