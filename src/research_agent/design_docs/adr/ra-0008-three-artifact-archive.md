# ADR ra-0008 — Three-artifact archive with a run-scoped ledger

## Context

What should outlive a run? Keeping everything turns the archive into a second database; keeping only the report loses the contract it was measured against.

## Decision

Exactly three **frozen, hashed artifacts** per run — Brief, Rubric, Report — each with a Markdown rendering, in the commons `DirectoryArtifactStore`. One **index row** per run (id, archetype, status, spend, unmet items, paths) goes to episodic memory, and the report is remembered as a semantic fact. The **Evidence Ledger is discarded**: it is run-scoped working state, and the report's citations carry URL, span, page hash and retrieval time for re-checking.

## Consequences

- `make runs ra` lists runs straight from the store and re-verifies every artifact's hash.
- Procedural memory is not built (`memory.procedural: false`): the agent learns nothing procedural across runs by design.

## Alternatives rejected

- **Persist the ledger** — duplicates the citations and grows without bound.
- **Only the report** — cannot show what was agreed.
