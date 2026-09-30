# ADR ra-0004 — Partial delivery on budget exhaustion

## Context

An autonomous run that hits its budget can fail, silently deliver what it has, or deliver it honestly. The user approved a contract; they are owed an account of how much of it was met.

## Decision

Budget exhaustion is never an error and never `COMPLETE`: the run synthesizes from what the ledger holds and delivers **`PARTIAL`** (`_final_status`: exhausted ⇒ PARTIAL, whatever the verdicts say). **Coverage & Limits** is built in code (`synthesis/limits.py`) from verdicts: for each unmet must-pass item, what it required, what was tried (queries, sources examined) and why the run stopped. The same applies when evidence simply does not exist (thin coverage).

## Consequences

- `budget_exhaustion_partial` and `unmet_must_pass_in_limits` are hard invariants; `ds-budget-exhausted` and `ds-thin-coverage` exercise them.
- A partial report is useful and truthful rather than absent.

## Alternatives rejected

- **Fail the run** — throws away paid-for evidence.
- **Let the model write the limits** — models understate gaps; this is exactly what the planted `false_coverage` flaw tests.
