# observability — a turn, as the numbers SLOs watch

`TurnAccountant` turns a finished turn into a `TurnOutcome` — success,
escalation, unauthorized action, hallucination, tool failure — from the
turn's decisions, and settles it into the platform's SLO trackers.

## Vocabulary

- **TurnOutcome** — the per-turn facts the SLO targets are measured on

## Public interface

- `TurnAccountant`, `TurnOutcome` (`accounting.py`)

## Configuration

The targets live in `observability.slo_targets` (platform).

## Collaborators

- `agentic_commons.observability.SLORegistry`.

## Episodes

E37 (SLOs and error budgets).

## Tests

Run with `./venv/bin/pytest tests/unit/test_design_gaps.py -q` (or `make test` for everything).
