# persistence — this agent's tables

The schema (`schema.sql`) and repositories. Money is integer cents; customer-facing lookups are scoped to the authenticated customer. Start-up refuses an unseeded database (`make dummy-data csa`).

- **Entry points:** `ensure_schema`, `require_seeded`, `OrdersRepository`, `UsersRepository`, `RefundLedgerRepository`
- **Configuration:** `persistence.db_path`
- **Design:** [ADR C-07](../../../design_docs/ADR.md#c-07--the-agent-database-is-built-from-its-seed)
