# ADR ra-0024 — Lightning writes, ultra judges

**Status:** accepted 2026-10-12 · amends ra-0018 (live by default) and the 2026-10-11
Producer-Critic judge line-up · see platform ADR 0013

## Context

Every research role led with nemotron-ultra (550B) and fell back to lightning; the Independent
Audit's judge led with lightning so that it would differ from the writer. The platform now
keeps ultra — the strongest model on the account — for judging, everywhere: the eval graders
and this agent's audit.

## Decision

- Every writing role (interviewer, planner, researcher, extractor, reflector, synthesizer,
  answerer) runs on lightning with thinking off, backed by super-120b (NVIDIA models only). Ultra is in none of
  their chains.
- The audit judge leads with ultra, backed by super-120b — never lightning, so no backup can
  make the writer grade itself. `audit.independent_judge` still holds start-up to that.
- The audit's `LlmJudge` is now a subclass of the platform's `agent_platform.judge.LlmJudge`:
  same prompt, same `CheckOutcome`, same "not judged" — now the skeleton's abstain.

## Consequences

- Research and synthesis quality may drop with a smaller writer; a live `/research` smoke run
  is the check, and the judge is now stronger than the writer it audits.
- Thinking is off for every writing role (with it on, lightning narrates its reasoning into
  the answer); the judge runs with ultra's default reasoning.
