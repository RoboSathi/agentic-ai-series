# ADR ra-0002 — Deterministic rubric generation, then user edits

## Context

Acceptance criteria written by a model in a second interview would be different on every run and could quietly drop what the user asked for. The criteria must be predictable from the Brief so the same scope always yields the same contract.

## Decision

The rubric is generated **deterministically** from `archetype template + Brief` (`rubric/generator.py`, `rubric/templates.yaml`) — no model call, no second interview — and the user then edits it (`/rubric add | drop | set`). Parameterisation rules, each tested: every user criterion → at least one item (including the explainer's criteria, via a judge item); every known unknown → an "acknowledged and bounded" item; source policy → source checks; recency → a recency check; and on every archetype one item requiring **at least one cited finding**, so no item set can pass vacuously. Each item carries `id, text, check(kind, name, params), threshold, must_pass, origin, maps_to`.

## Consequences

- Same Brief, same rubric, same hash.
- The oracle's `scope_maps_to_rubric` invariant is checkable because the mapping is explicit (`maps_to`).
- The findings item was added after the evals caught a decision scope with no criteria being delivered as an empty `COMPLETE` report.

## Alternatives rejected

- **LLM-written rubric** — non-reproducible; cannot be gated on.
- **Fixed rubric per archetype** — ignores the user's own criteria.
