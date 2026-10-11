"""Demo dataset for the customer support agent.

Schema (schema.sql), generated seed data (seed_data.sql), and the built
SQLite database are owned by this agent. The actual database path is the
resolved `persistence.db_path` (the agent overlays it) from the agent configuration, so dummy-data
and runtime always open the same file.
"""

from pathlib import Path

DUMMY_DATA_DIR = Path(__file__).resolve().parent
# The generated seed `make dummy-data csa` writes and builds the database from.
SEED_DATA_FILE = "seed_data.sql"
