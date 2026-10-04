# console — this agent in the operator shell

`CustomerSupportHost` implements the platform's `AgentHost`: the console's
title, how to build the agent, the stage labels only this agent emits, and
its own slash commands — `/queue` (human review: list, approve, deny,
staffing maths), `/void` (list issued refunds, or void one before it
settles), `/settle` (settle one, or `all`, now) and `/kb` (the knowledge
index). As a `HousekeepingHost` it settles refunds whose void window has
closed, every `settlement.sweep_interval_seconds`.

## Vocabulary

- **host** — what the platform console runs, named in `agent.host`

## Public interface

- `CustomerSupportHost` — `prepare` (warms the intent classifier and NLI model before the event loop, syncs the knowledge base, recovers orphaned runs), `build`, `commands`, `stage_labels`, `housekeeping`
- `QueueCommand`, `VoidCommand`, `SettleCommand` (`commands.py`)

## Configuration

`agent.host: customer_support_agent.console:CustomerSupportHost` in this agent's config.

## Seams

- The platform resolves the host from config; nothing in `agent_platform` imports this package.

## Collaborators

- `graph.build_agent`, `graph.build_settlement_service`, `review.ReviewQueueService`, `memory` (knowledge base).

## Episodes

E38 (the human review queue).

## Tests

Run with `./venv/bin/pytest tests/unit/test_commands.py tests/unit/test_tui.py -q` (or `make test` for everything).
