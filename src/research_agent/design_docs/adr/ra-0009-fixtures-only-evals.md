# ADR ra-0009 — Fixtures-only evals

## Context

Evals that touch the live web or a hosted model are neither reproducible nor free, and an invariant that holds "usually" cannot gate at 100%.

## Decision

Evals run against a **synthetic fixture world** recorded through the real adapters and **replayed** (a miss raises), with the fixture world's **reference model** in every role. The oracle gates at 100%; the security suite is native (hostile pages in the fixtures); the judge stage uses an offline reference judge under the fixture backend. Ablations are config overlays over the same fixtures.

## Consequences

- Every run is exact and fast (18 golden cases in ~20 s), so `make eval ra` can run in CI.
- What is measured is the agent's machinery, not a hosted model's quality; `model_backend: routed` is the switch for the latter.

## Alternatives rejected

- **Live evals** — flaky and billed.
- **Cassette-style HTTP recording** — pins bytes, not meaning; a fixture world has known ground truth.
