# ADR 0006 — Retire `episodes/` and `design_patterns/`

- Status: accepted
- Date: 2026-10-06

## Context

The repository carried two teaching layers beside the agents:

- `episodes/eNN/app.py` — about 45 small scripts, one per episode, most
  running against an offline scripted model and hard-coded toy data.
  They were already outside the gates (`--ignore=tests/episodes`, dropped
  from lint and types in ADR 0005), only five had been converted to the
  config-driven shape, and the episode list itself is being re-planned.
- `design_patterns/` — one reference implementation per WF/AG pattern,
  plus the catalogue and `make patterns` coverage matrix (ADR 0001).

Both agents (`customer_support_agent`, `research_agent`) now implement
almost every pattern the series teaches, in production form, and every
harness feature can be moved per agent through its config overlay,
profiles, ablations and `/harness`. A separate copy of each idea drifts
from the code the viewer is meant to trust, and teaches a shape that no
running product has.

## Decision

Delete `episodes/` and `design_patterns/` outright, with their tests,
`make episode`, `make patterns` and the generated coverage matrix.

An episode is demonstrated by running an agent under the config that
episode needs. Each agent's `design_docs/patterns.md` records where it
uses each pattern (carried over from the `patterns.yaml` files `make
patterns` read).

PRD FR-58 and FR-59 are retired, not silently dropped.

## Consequences

- One code path per idea: what is shown on screen is the code that runs.
- `fixed_loop` has no implementation in either agent. The budgeted loops
  in the research agent are not fixed loops — the model or reflection
  decides when they stop, and the budget only caps them. A genuine fixed
  loop (count known before it starts) is left for a later decision;
  pass^k reliability runs in the evals framework are the leading candidate.
- Platform code that existed only for episodes becomes dead and is removed
  when it has no remaining caller.

## Alternatives rejected

- **Keep the catalogue and `make patterns`.** The matrix's value was
  cross-agent gap finding; with two agents, the two `patterns.md` files
  answer the same question without a tool to maintain.
- **Rewrite the episodes as config overlays.** The episode list is
  changing; overlays will be written per episode as each is recorded,
  against the agents, not ahead of the plan.
