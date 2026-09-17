"""Schema initialization for the customer support agent's database.

The DDL itself lives in customer_support_agent/dummy_data/schema.sql — the
single source of truth, used both by generate_dummy_data.py to build the
demo database and by ensure_schema() here to initialize any fresh database
(e.g. a test's temporary SQLite file) at runtime.
"""

from agentic_commons.persistence.database import Database
from customer_support_agent.dummy_data import DUMMY_DATA_DIR

_SCHEMA_FILE = DUMMY_DATA_DIR / "schema.sql"


def ensure_schema(database: Database) -> None:
    """Create the users/orders/hitl_queue tables if they don't already exist."""
    database.execute_script(_SCHEMA_FILE.read_text())
