# ADR 0002 — The agent registry: `make <target> <agent>`

## Context

The Makefile exported `AGENTIC_OVERLAYS := src/customer_support_agent/…`
unconditionally, so every target was the customer-support agent's. With a
second agent, a silent default would mean running one agent's evals while
believing you ran the other's.

## Decision

A registry at the top of the Makefile: `AGENTS := csa ra` and, per agent,
`<name>_PKG`, `<name>_OVERLAY`, `<name>_DATA`, `<name>_RUNS`. Agent-scoped
targets take the agent as their **second goal** (the no-op-goal trick the
file already used for `refund-limit`): `make tui ra`, `make eval csa`,
`make eval-diff ra A=… B=…`. The name alone sets `AGENTIC_OVERLAYS`.

- A missing, unknown or doubled name prints usage and exits 2.
- Repo-wide targets (`install`, `lint`, `type-check`, `format`, `test`,
  `verify`, `episode`, `llm-check`, `patterns`) take no name.
- csa-only admin targets (`show-memory`, `dream`, `clean-memory`,
  `refund-limit`) keep their names and pin csa's overlay themselves.
- `verify` imports each agent's composition root **in its own process**
  (importing both in one process lets the first overlay declaration win).
- `ABLATION=<name>` appends `src/<pkg>/evals/ablations/<name>.yaml` as one
  more config layer (ra's ablations).
- `make data <agent>` replaces `dummy-data`: csa rebuilds its database, ra
  records and validates the fixture world.

## Consequences

- `make tui csa` and `make eval-oracle csa` run exactly the commands
  `make tui` / `make eval-oracle` ran before (`tests/test_makefile_registry.py`).
- Adding an agent is four registry lines; nothing in the platform changes.
- Muscle memory breaks once: `make tui` now asks which agent.

## Alternatives rejected

- **`AGENT=ra make tui`.** Forgettable, and a stale shell variable silently
  picks the agent.
- **Default to csa.** Exactly the silent mis-selection the change exists to
  remove.
- **One Makefile per agent.** Duplicates every platform target.
