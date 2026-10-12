"""Creating this agent's tables from the single source of truth.

The DDL lives in `schema.sql` beside this module and is read from disk
rather than duplicated here: two copies of a schema drift, and the copy
in the test fixture is always the one that drifts.

A database built before the platform's one review queue (ADR 0012) has a
`hitl_queue` table of its own. The first time this version opens it, its
rows are moved into the review queue as items, and the table is dropped.
"""

import hashlib
from pathlib import Path

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.core.primitives.money import Money
from agent_platform.review import ReviewQueue
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import HitlRequest, ReplyReview
from customer_support_agent.persistence.repositories import HitlQueueRepository, UsersRepository

_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
# Which seed the database was built from (`make dummy-data` writes it; start-up checks it).
_SEED_TABLE = "seed_info"
# The column a queue built by this version has and an older one lacks.
# The table escalations lived in before the platform's review queue.
_OLD_QUEUE = "hitl_queue"
# Order columns added after databases were already built. `CREATE TABLE IF NOT
# EXISTS` leaves an existing table as it was, so each is added in place, once.
_ADDED_ORDER_COLUMNS = {"carrier_note": "TEXT", "refund_settles_at": "TEXT"}
# Refund statuses no code path produces any more, and what an older row means now.
# `approved` was only ever seeded: a reviewer's approval issues the refund, so a seeded
# approved refund is one that has paid out and settled.
_RETIRED_REFUND_STATUSES = {"approved": "completed"}


class AgentDatabaseNotSeededError(RuntimeError):
    """The agent's own database has its tables but no customers in them."""


def require_seeded(pool: SqliteConnectionPool, path: Path, seed_sql: str | None = None) -> None:
    """Refuse a database with no customers, or one built from a different seed.

    `ensure_schema` makes an empty file a valid database, so a deleted one comes
    back as tables with nobody in them: the agent would start, look healthy, and
    reject every customer. A database built from an older `seed_data.sql` is the
    same failure in disguise: the codes a person reads in the seed are not the
    ones the database holds, and every one is "not accepted". Only the agent's
    own file is checked; a pool a caller injects is that caller's fixture.
    """
    settings = get_agent_settings()
    if UsersRepository(pool).count() == 0:
        raise AgentDatabaseNotSeededError(settings.unseeded_database.format(path=path))
    if seed_sql is not None and built_from(pool) != seed_digest(seed_sql):
        raise AgentDatabaseNotSeededError(settings.stale_database.format(path=path))


def seed_digest(seed_sql: str) -> str:
    """What identifies a seed: a digest of its text."""
    return hashlib.sha256(seed_sql.encode("utf-8")).hexdigest()


def record_seed(pool: SqliteConnectionPool, seed_sql: str) -> None:
    """Note in the database which seed it was built from."""
    pool.execute_script(f"CREATE TABLE IF NOT EXISTS {_SEED_TABLE} (digest TEXT NOT NULL);")
    pool.execute(f"DELETE FROM {_SEED_TABLE}")
    pool.execute(f"INSERT INTO {_SEED_TABLE} (digest) VALUES (?)", (seed_digest(seed_sql),))


def built_from(pool: SqliteConnectionPool) -> str:
    """The digest of the seed this database was built from ("" when it does not say)."""
    present = list(pool.iter_rows(f"PRAGMA table_info({_SEED_TABLE})"))
    row = pool.query_one(f"SELECT digest FROM {_SEED_TABLE}") if present else None
    return str(row["digest"]) if row else ""


def ensure_schema(pool: SqliteConnectionPool) -> None:
    """Create the domain tables from the single source of truth, upgrading older tables."""
    _add_missing_columns(pool, "orders", _ADDED_ORDER_COLUMNS)
    _retire_refund_statuses(pool)
    pool.execute_script(_SCHEMA_FILE.read_text())
    # The platform's review queue, where this agent's escalations are items (ADR 0012).
    ReviewQueue(pool, agent_id=get_settings().agent.id)
    _move_old_queue(pool)


def _move_old_queue(pool: SqliteConnectionPool) -> None:
    """Move an older database's `hitl_queue` rows into the review queue, then drop it."""
    present = {row["name"] for row in pool.iter_rows(f"PRAGMA table_info({_OLD_QUEUE})")}
    if not present:
        return
    queue = HitlQueueRepository(pool)
    for row in pool.iter_rows(f"SELECT * FROM {_OLD_QUEUE} ORDER BY id"):
        cents = row.get("requested_amount_cents")
        detail = row.get("detail")
        queue.save(
            HitlRequest(
                order_id=row.get("order_id"),
                user_id=int(row["user_id"]),
                requested_amount=None if cents is None else Money.from_cents(int(cents)),
                action=row.get("action") or "refund",
                status=row["status"],
                created_at=str(row["created_at"]),
                session_id=row.get("session_id"),
                detail=None if not detail else ReplyReview.model_validate_json(detail),
                reason=str(row.get("reason") or ""),
            )
        )
    pool.execute_script(f"DROP INDEX IF EXISTS idx_hitl_status; DROP TABLE {_OLD_QUEUE};")


def _add_missing_columns(pool: SqliteConnectionPool, table: str, columns: dict[str, str]) -> None:
    """Add any column this version declares and an older table lacks."""
    present = {row["name"] for row in pool.iter_rows(f"PRAGMA table_info({table})")}
    if not present:
        return
    for column, declaration in columns.items():
        if column not in present:
            pool.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _retire_refund_statuses(pool: SqliteConnectionPool) -> None:
    """Rewrite any order still carrying a refund status this version no longer has."""
    if not list(pool.iter_rows("PRAGMA table_info(orders)")):
        return
    for retired, current in _RETIRED_REFUND_STATUSES.items():
        pool.execute(
            "UPDATE orders SET refund_status = ? WHERE refund_status = ?", (current, retired)
        )
