# console — this agent in the operator shell

`CustomerSupportHost` implements the platform's `AgentHost`: the console's
title, how to build the agent, the stage labels only this agent emits, and
its own slash commands — `/queue` (human review: list, approve, deny,
staffing maths) and `/kb` (the knowledge index).

## Vocabulary

- **host** — what the platform console runs, named in `agent.host`

## Public interface

- `CustomerSupportHost` — `prepare` (warms the intent classifier and NLI model before the event loop, syncs the knowledge base, recovers orphaned runs), `build`, `commands`, `stage_labels`
- `QueueCommand` (`commands.py`)

## Configuration

`agent.host: customer_support_agent.console:CustomerSupportHost` in this agent's config.

## Seams

- The platform resolves the host from config; nothing in `agentic_commons` imports this package.

## Collaborators

- `graph.build_agent`, `review.ReviewQueueService`, `memory` (knowledge base).

## Episodes

E38 (the human review queue).

## Tests

Run with `./venv/bin/pytest tests/unit/test_commands.py tests/unit/test_tui.py -q` (or `make test` for everything).
