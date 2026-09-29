# persistence — this agent's tables

The DDL (`schema.sql`: users, orders, the review queue, and the platform
tables) and the repositories that read and write them. Money is integer
cents throughout. Customer-facing lookups are always scoped to the
authenticated customer.

## Vocabulary

- **user_code** — the credential a customer types — deliberately not the enumerable `user_id`
- **refunds ledger** — what was actually paid, separate from an order's current `refund_status`

## Public interface

- `ensure_schema(pool)` (`schema.py`)
- `OrdersRepository`, `UsersRepository`, `HitlQueueRepository`, `RefundLedgerRepository` (`repositories.py`)

## Configuration

`persistence.db_path` (the agent overlays it).

## Seams

- Repositories take an injected pool; tests and evals pass their own.

## Collaborators

- `domain`, `tools`, `review`, `resilience`, `evals.fixture`.

## Episodes

E01, E35, E38, E48b, E48c.

## Tests

Run with `./venv/bin/pytest tests/unit/test_authentication_and_scope.py tests/unit/test_basic_sanity.py -q` (or `make test` for everything).
