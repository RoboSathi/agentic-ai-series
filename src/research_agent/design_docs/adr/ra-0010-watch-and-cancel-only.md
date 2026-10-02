# ADR ra-0010 — Watch and cancel only

## Context

Once the contract is approved the run is autonomous. Mid-run steering would make the contract meaningless — the audit could no longer say what the run was held to.

## Decision

The user can **watch** (live stage view and side panel: waves, budget meters, rejected sources, audit verdicts) and **cancel** (the host's cancel key, `ctrl+x`, or `/cancel`). Cancel is cooperative: workers check before every search/fetch, the wave router between waves. The run ends `CANCELLED`, its checkpoint kept.

## Consequences

- No pause, no steering, no contract edits after `/approve`: a new scope is a new contract.
- The cancel binding is a priority binding so it works while the input box has focus.

## Alternatives rejected

- **Pause/resume with edits** — breaks the frozen contract.
- **Hard kill** — loses the checkpoint and the audit trail.
