# dummy_data — the demo shop

`make dummy-data` generates a deterministic synthetic shop from
`generator.yaml` (100 users, each a different famous scientist, mathematician or
computer scientist; 1000 orders, every one starting with refund status `none`;
fixed seed; dates are relative to now) into `seed_data.sql`, rebuilds `agentic.db` from the schema, seed and
`agent_policy_seed.sql`, logs the policy it reads back from the database,
resets checkpoints and episodic logs that remember the old orders
(`KEEP_MEMORY=1` keeps them), and rebuilds the eval fixture. It never
touches `data/evals.duckdb` or the observability store.

## Vocabulary

- **generator spec** — `generator.yaml`: counts, seed, status weights, catalogue, carriers, locations
- **agent policy seed** — role, authority rank, permissions and budgets (`$50` per action, `$4,000`/24h window)

## Public interface

- `generate_seed_sql()`, `build_agent_database(seed_sql)`, `reset_derived_state()`, `refund_status_for(rng, status)`, `main(argv)` (`generate_dummy_data.py`)

## Configuration

`customer_support_agent.dummy_data_generator_path`, `db_path`; user-code length and alphabet from the same section.

## Collaborators

- `persistence/schema.sql`, `evals.fixture` (the eval golden set uses its own time-stable fixture, not this data).

## Episodes

Every episode runs against this data.

## Tests

Run with `./venv/bin/pytest tests/integration/test_eval_pipeline.py::TestDummyData tests/unit/test_refund_eligibility.py -q` (or `make test` for everything).

## Status and limits

`agentic.db`, `checkpoints.db`, `knowledge.db` and `episodic/` are generated and gitignored.
