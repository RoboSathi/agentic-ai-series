"""This agent's platform store, opened the one way every admin command opens it."""

from agentic_commons.config import get_settings
from agentic_commons.persistence import SqliteConnectionPool, SqlitePlatformStore
from customer_support_agent.config import get_agent_settings


def agent_store() -> SqlitePlatformStore:
    """The agent's own database, scoped to its configured identity."""
    pool = SqliteConnectionPool(get_agent_settings().resolved_db_path)
    return SqlitePlatformStore(pool, agent_id=get_settings().agent.id)
