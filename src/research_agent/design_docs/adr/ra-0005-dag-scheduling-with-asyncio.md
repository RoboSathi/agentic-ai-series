# ADR ra-0005 — DAG scheduling with asyncio, as LangGraph supersteps

## Context

The planner produces sub-questions with dependencies. Running them one by one wastes wall-clock; running them all at once ignores dependencies and budgets. LangGraph should own state and checkpoints.

## Decision

The plan is a DAG validated by the commons `TaskGraph` (acyclic, closed, bounded by `max_dag_nodes`, every node traced to a rubric item). Each **wave** is a LangGraph superstep: `wave_start` computes the ready set and fans out one `Send` per node; `research_node` runs under the commons `AsyncDagScheduler`'s semaphore (`budget.concurrency`); `wave_done` joins. Budgets are reserved synchronously before every await.

## Consequences

- A checkpoint at every wave boundary; a crash resumes at the wave it interrupted.
- Dependents of a failed node are blocked, not run on missing inputs.
- Concurrency is a budget field, visible in the contract.

## Alternatives rejected

- **Hand-rolled asyncio loop outside LangGraph** — loses checkpoints and the platform's durability.
- **Sequential execution** — simple, slow, and hides nothing worth hiding.
