# ADR ra-0019 — The fixture world and its evals are removed

**Status:** accepted 2026-10-02 · supersedes ra-0009 and ra-0015

## Context

The synthetic Kelvari world (`fixture_world/`), its reference model and the `demo` profile
existed so the agent could run offline (ra-0015) and its evals could be deterministic
(ra-0009). The agent now runs live by default (ra-0018); the demo posture served no
purpose a viewer of the series needs, and kept a second, fake "model backend" switch in
product code.

## Decision

Delete `fixture_world/`, `config/profiles/demo.yaml`, the `model_backend` setting and every
branch on it, and the eval suite built on the world (`evals/`: plugin, oracle, golden and
security cases, planted flaws, ablations, reference judge). `make data ra` and `make eval ra`
now say the agent has neither. Tests inject scripted models through
`build_agent(clients=...)` rather than a product-side mode.

## Consequences

- The research agent has **no eval suite** until one is rebuilt over live or recorded web
  data. The platform's eval pipeline is unchanged (the support agent still uses it).
- About 97 offline flow tests that drove whole runs through the reference model are gone;
  unit tests of each stage remain.
- Git history keeps the removed code (`git log -- src/research_agent/fixture_world`).
