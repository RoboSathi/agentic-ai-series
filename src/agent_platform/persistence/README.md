# persistence — SQLite access and platform tables

Thread-safe SQLite access and the platform's own per-agent tables (agent
identity, permissions, budgets, gatekeeper ledger). An agent's business
tables and repositories live in the agent, not here.

## Vocabulary

- **platform store** — the platform's tables inside an agent's database, scoped by `agent_id`

## Public interface

- `SqliteConnectionPool` — one shared connection under a lock; `execute`, `execute_many`, `execute_script`, `query_one`, `iter_rows` (`database.py`)
- `PlatformStore` (Protocol), `SqlitePlatformStore`, `as_platform_store` (`platform.py`)
- `Repository` (Protocol), `ExecutionResult`, `IN_MEMORY`

## Configuration

Section `persistence:` — `db_path` (a neutral default; each agent overlays its own).

## Seams

- A pool is injectable everywhere (`build_agent(pool=...)`); tests use `:memory:`.

## Collaborators

- Used by `audit`, `gatekeeper`, `memory`, `durability`, `security.auth` and every agent repository.

## Episodes

E01 onward.

## Tests

Run with `./venv/bin/pytest tests/unit/test_audit_persistence_llm.py -q` (or `make test` for everything).

## Status and limits

Always on.
