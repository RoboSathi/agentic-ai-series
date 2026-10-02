# ADR ra-0003 — Configurable audit independence

## Context

A verifier that shares the writer's context or model shares its blind spots. The independent audit must see only what an outside auditor would, and its judge must not be the writer — but deployments may lack a second model, and the effect of independence should be measurable.

## Decision

The Independent Audit runs on the commons `IsolatedGraphRunner` and its entry point accepts only `(contract, report, ledger_snapshot)`; any other input key is refused (`UndeclaredInputError`). Checks run cheapest first: deterministic → NLI → judge. The judge is the `judge` role in `llm.roles`; while `research_agent.audit.independent_judge` is true, start-up refuses a judge that resolves to the synthesizer's model (`RoleCollisionError`). Independence is a setting, recorded on every audit result.

## Consequences

- The `same_model_judge` ablation measures what independence buys: the judge-only planted flaw (trade-offs omitted) goes uncaught.
- Deterministic and NLI checks do not depend on the judge at all, so most flaws are caught either way.

## Alternatives rejected

- **Audit inside the run graph's state** — the auditor could read the run's scratch state.
- **Always require a second model** — blocks single-model deployments rather than stating the trade-off.
