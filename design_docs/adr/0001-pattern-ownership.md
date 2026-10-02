# ADR 0001 — Pattern ownership: composition is agent-owned

## Context

The series teaches sixteen WF/AG patterns (`design_patterns/catalogue.py`)
plus the LLD and resilience patterns. The research agent composes several
of them — a planner, a ReAct-style worker, reflection, an
evaluator-optimizer loop around an independent audit — and the obvious
temptation is to "promote" each into `agentic_commons` so the next agent
can reuse it. The customer-support agent composes a different set
(tool-use loop, router, quality gate). The two compositions look alike in
a diagram and share almost no code.

## Decision

Pattern **composition** lives inside the agent's own graph. Only
domain-free **mechanisms** — things with no opinion about what the agent
is for — go in `agentic_commons`: a DAG scheduler, a budget guard, an
isolated sub-graph runner, a role router, an evidence ledger, a web
gateway, an acceptance-check registry.

A pattern is promoted to commons only when a **second agent needs the
identical thing**, and then as a mechanism with a Protocol, not as the
first agent's node code moved sideways.

`design_patterns/` stays what it is: the teaching layer and the catalogue,
one runnable reference implementation per pattern, independent of either
product. `make patterns` shows which catalogue patterns the products use
and where.

## Consequences

- ra's planner, reflector and Independent audit are ra code; the scheduler
  they run on, the isolation they rely on and the budget that bounds them
  are commons.
- A third agent gets mechanisms, not a framework — it composes its own.
- The coverage matrix, not code sharing, is how the series shows the same
  pattern in two products.

## Alternatives rejected

- **Promote each pattern to commons as a reusable node.** Every node would
  carry the first agent's state shape and vocabulary, which is exactly the
  back-dependency the platform forbids.
- **Build ra on `design_patterns/`.** Those are single-file teaching
  implementations; production code depending on them would freeze the
  teaching layer.
