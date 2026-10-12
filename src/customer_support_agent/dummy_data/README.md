# dummy_data — the demo shop

`make dummy-data csa` generates a deterministic shop from `generator.yaml` (100 users, 1,000 orders, fixed seed) into `seed_data.sql`, rebuilds `db/customer_support_agent/agent.db` with the policy seed, resets derived state (`KEEP_MEMORY=1` keeps memory), and rebuilds the eval fixture.

- **Entry points:** `generate_dummy_data.py` (`main`), `generator.yaml`, `seed_data.sql`, `agent_policy_seed.sql`
- **Configuration:** `customer_support_agent.dummy_data_generator_path`
- **Design:** [ADR C-07](../../../design_docs/ADR.md#c-07--the-agent-database-is-built-from-its-seed)
