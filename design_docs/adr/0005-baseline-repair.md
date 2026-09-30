# ADR 0005 — Baseline repair before the research agent

- Status: accepted
- Date: 2026-09-30

## Context

The research-agent build spec requires `make verify` to be green at the
end of every phase. On `main` (b65a408) it was red on every gate:

- `ruff format --check`: 56 files unformatted;
- `mypy --strict`: 44 errors in 25 files (commons, csa, episodes,
  design_patterns), several of them real runtime defects;
- `pytest`: two modules failed to import, which aborted collection; once
  they imported, 143 tests failed and 24 errored — all stale against APIs
  the author had since refactored (`GatekeeperImpl(policy=...)`,
  `WorkingMemory.write` removed, intent labels renamed,
  `StageLabelRegistry.label_for` removed, `NamespacedToolRegistry.invoke`
  removed, …); coverage 78%.

## Decision

A single behaviour-preserving repair commit, before any feature work:

1. **Episodes are outside the gates** (owner's decision: they will be
   rewritten). `episodes/` is dropped from `lint`, `format`, `type-check`,
   `verify` and coverage; `tests/episodes` is ignored by pytest
   (`pyproject.toml addopts`). `make episode` still runs one, with the csa
   overlay named explicitly.
2. **Stale tests are deleted** (owner's decision). Exactly the test
   functions that failed or errored on the baseline were removed
   (a parametrised function whenever any of its cases failed); classes or
   files left with no tests were removed. 152 functions; git history keeps
   them. Two import-only fixes kept tests that were otherwise current
   (`is_customer_ready` moved to `presentation.reporting`; the
   `PROMPT_ARTIFACT` tests covered a removed feature).
3. **Formatting**: `ruff format` over `src/ tests/ design_patterns/`.
4. **Type-only fixes** (no runtime change): gatekeeper `_eval_*` return
   types, `Literal` narrowing in entailment/gatekeeper/screening/guardrail
   fields, a `MemoryReporter.emit(quiet=)` protocol parameter that the
   implementation already had, `AgentPolicy` narrowing in csa's agent,
   `order_validator` map typing, and a dropped `roles=` kwarg that
   `AuthorityContext` has never had (pydantic ignored it at runtime).
5. **Runtime defects fixed because typing exposed them**:
   - `GatekeeperImpl._eval_cumulative` returned a 3-tuple on a
     misconfigured policy, which the caller unpacks as a pair (a crash);
   - csa `act.py` wrote `AuditEntry(decision=report.reason_code)` for an
     unverified side effect, which pydantic rejects (a crash on exactly
     the path meant to record a failure) — now `decision="deny"` with the
     reason code prefixed to the rationale;
   - `design_patterns/agentic/{tool_use,react}.py` called the removed
     `registry.invoke`; the teaching layer now has its own
     `_invoke.invoke_tool` (patterns are shown without the Gatekeeper on
     purpose).
6. **Dead module removed**: `agentic_commons/gatekeeper/automation.py`
   imported `AutomationAssessment`, which commit 633fc9e deleted along
   with the Sheridan decision; nothing in `src/` imported it.

## Consequences

- `make verify` is green on the repaired baseline, so the phase rule can
  be honoured from phase 1.
- The `gatekeeper.automation` and `gatekeeper.surface_policy` config
  sections now have no consumer in `src/` (only episodes referenced the
  policy classes). They are left in place for the episode rewrite and
  flagged here as config-without-implementation to resolve then.
- csa's behaviour is unchanged except on two paths that previously
  crashed.

## Alternatives rejected

- *Quarantine the stale tests* — declined by the owner in favour of deletion.
- *Rewrite the stale tests* — would encode inferred intent into csa's
  specification; declined.
- *Scope verify to new code only* — would leave the DoD gate meaningless.
