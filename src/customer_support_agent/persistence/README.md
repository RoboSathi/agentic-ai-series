# persistence — this agent's tables

The DDL (`schema.sql`: users, orders, the review queue, and the platform
tables) and the repositories that read and write them. Money is integer
cents throughout. Customer-facing lookups are always scoped to the
authenticated customer.

## Vocabulary

- **user_code** — the credential a customer types — deliberately not the enumerable `user_id`
- **refunds ledger** — what was actually paid, separate from an order's current `refund_status`

## Public interface

- `ensure_schema(pool)`, `require_seeded(pool, path)`, `AgentDatabaseNotSeededError` (`schema.py`)
- `OrdersRepository`, `UsersRepository`, `HitlQueueRepository`, `RefundLedgerRepository` (`repositories.py`)

## Configuration

`persistence.db_path` (the agent overlays it); `customer_support_agent.unseeded_database`
(the start-up refusal's wording).

The database is gitignored and built by `make dummy-data csa`. `ensure_schema` turns a
missing file into empty tables, so the composition root also calls `require_seeded` on
the agent's own database: with no customers it refuses to start rather than answer every
code as unknown. To stop versioning a database, untrack it with `git rm --cached`: a plain
`git rm` deletes the local file too, which is how an unseeded database once reached a
running console.

## Seams

- Repositories take an injected pool; tests and evals pass their own.

## Collaborators

- `domain`, `tools`, `review`, `resilience`, `evals.fixture`.

## Episodes

E01, E35, E38, E48b, E48c.

## Tests

Run with `./venv/bin/pytest tests/unit/test_authentication_and_scope.py tests/unit/test_deterministic_checks.py -q` (or `make test` for everything).
