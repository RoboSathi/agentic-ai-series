"""Demo dataset for the customer support agent.

Schema (schema.sql), generated seed data (seed_data.sql), and the built
SQLite database (agentic.db, gitignored) all live here — self-contained,
not under agentic_commons/data.
"""

from pathlib import Path

DUMMY_DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DUMMY_DATA_DIR / "agentic.db"
