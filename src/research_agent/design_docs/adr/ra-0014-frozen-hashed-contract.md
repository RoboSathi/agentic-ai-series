# ADR ra-0014 — The frozen, hashed contract

## Context

The autonomous phase must be held to exactly what the user approved, and anyone reading the report later must be able to tell what that was.

## Decision

At `/approve` the Brief, Rubric and Budget are frozen into a `Contract` whose **canonical hash** (commons `canonical_hash`) is recorded in the audit ledger and the stage feed. The run graph receives only the contract. At the end the hash is re-verified (`contract.verified` audit event) and the Brief and Rubric are archived as frozen artifacts.

## Consequences

- `contract_hash_unchanged` is a hard invariant.
- A Brief may lower budgets, never raise them above the deployment's ceiling (`Budget.capped_by`).

## Alternatives rejected

- **Mutable plan state** — the audit could not say what the run was held to.
