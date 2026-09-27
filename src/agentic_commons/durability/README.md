# durability — runs that survive failure

Checkpointed execution for LangGraph agents. Every turn runs as its own
thread; if a node fails, state is saved and the run can resume from that
node without repeating side effects already committed. A human-in-the-loop
pause is the same mechanism: the graph interrupts, waiting on a decision,
and resumes with it.

## Vocabulary

- **thread** — one turn's durable run, `<session>:<id>`
- **resumable** — failed part-way with state checkpointed
- **interrupted** — paused on purpose, waiting on a decision (e.g. a reviewer)

## Public interface

- `create_checkpointer(state_types=..., path=...)` → `SqliteCheckpointer` (encrypted, sync SQLite bridged for async)
- `DurableGraphRunner` — `start`, `resume(thread, decision)`, `values`, `history`
- `RunRegistry` — run records, `resumable`, `recover_orphans`; `purge_expired`
- `DurableAgent` (Protocol), `ResumableRunError`, `ResumeResult`, `RunOutcome`

## Configuration

Section `durability:` — `db_path`, `mode`, `retention_days`, `preview_chars`, `encryption`.
An agent overlays its own `db_path`. Env: `DURABILITY__DB_PATH`.

## Seams

- The composition root may inject a checkpointer (`build_agent(checkpointer=...)`) — the eval harness gives every case its own file.

## Collaborators

- Used by the agent graph; reviewers resume interrupted runs through the agent's `/queue approve|deny`.

## Episodes

E35 (durable execution), E38 (human review).

## Tests

Covered through the agent tests (`tests/unit/test_production_hardening.py`); dedicated durability tests are an open item.
