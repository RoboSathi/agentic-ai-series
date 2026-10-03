# ADR ra-0015 — Model backend: fixture or routed (deviation)

**Status:** superseded by ra-0019 (2026-10-02) — the fixture world and its evals were removed.

## Context

The build brief requires `make agent ra` to run the whole flow in replay mode and evals to be deterministic, but this environment has no model credentials and hosted models are not deterministic.

## Decision

`research_agent.model_backend: fixture | routed`. **fixture** puts the fixture world's reference model behind every role (`RoleModelRouter` with `fixture_client_factory`); it reads the same prompts a hosted model would and answers each role's schema. **routed** resolves `llm.roles` through `RoleModelRouter` to pinned platform clients. The demo and eval posture defaults to `fixture` + `web.mode: replay`.

## Consequences

- The whole agent — scoping, planning, extraction, synthesis, audit — runs offline and reproducibly.
- The reference model is deliberately naive in documented ways (ADR ra-0007/ra-0003 ablations depend on it).
- Real-model quality is measured by switching to `routed`, not by these evals.

## Alternatives rejected

- **Scripted responses per test** — cannot drive the TUI demo or evals over arbitrary cases.
- **Require credentials** — not available here, and non-deterministic anyway.
