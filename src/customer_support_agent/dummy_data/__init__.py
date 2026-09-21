"""Demo dataset for the customer support agent.

Schema (schema.sql), generated seed data (seed_data.sql), and the built
SQLite database are owned by this agent. The actual database path is the
resolved `customer_support_agent.db_path` from the agent configuration, so dummy-data
and runtime always open the same file.
"""

from pathlib import Path

DUMMY_DATA_DIR = Path(__file__).resolve().parent
