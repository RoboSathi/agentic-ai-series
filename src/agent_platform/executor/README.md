# executor — the deterministic execution boundary

Runs a tool call the Gatekeeper has already approved, and nothing else:
timeout, idempotency, lifecycle events and an audit hook around the handler.
The model can propose; only an `ApprovedToolCall` reaches this module.

## Vocabulary

- **ApprovedToolCall** — a proposal that passed the Gatekeeper, with its authority context
- **ExecutionOutcome** — status (`completed`, `failed`, `timed_out`, …), whether it was replayed, duration, idempotency key

## Public interface

- `ToolExecutor` (Protocol), `ToolExecutorImpl` (`executor.py`)
- `ToolExecutionHandler`, `ExecutionAuditHook`, `ExecutionAuditEvent`, `ExecutionPhase`, `ExecutionStatus` (`types.py`)

## Configuration

No section of its own; the timeout comes from `resilience.timeout`.

## Seams

- The agent's tool handler is injected as a `ToolExecutionHandler`.

## Collaborators

- Called by the agent's `act` node after `gatekeeper`; writes to `audit`; replays through `resilience.idempotency`.

## Episodes

E02, E33 (the boundary between proposal and execution).

## Tests

Run with `./venv/bin/pytest tests/unit/test_gatekeeper.py -q` (or `make test` for everything).

## Status and limits

Always on.
