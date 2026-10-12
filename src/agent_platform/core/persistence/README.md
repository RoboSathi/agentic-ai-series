# core/persistence — SQLite access and platform tables

A thread-safe SQLite pool and the platform's own per-agent tables (identity, permissions, budgets, review queue, PII vault). Business tables and repositories live in each agent.

- **Entry points:** `SqliteConnectionPool`, `PlatformStore`, `SqlitePlatformStore`, `Repository`, `IN_MEMORY`
- **Configuration:** `persistence.db_path`
- **Design:** [LLD §A3](../../../../design_docs/LLD.md#a3-core-primitives-persistence)
