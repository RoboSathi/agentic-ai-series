"""The eval fixture: one seeded database, copied fresh for every case.

Built once into a template file (schema, the refunds ledger, the golden
seed, then this agent's runtime policy), and copied per case so no case
can see another's refunds. The same builder serves `make dummy-data` and
`make eval`, so the two can never disagree about the fixture.

The oracle reads the fixture's *facts* (who owns what, what each order
totals, what the limit is) straight from the seed, never through the
agent's own repositories — grading the agent with its own code would
check nothing.
"""

import json
import shutil
import sqlite3
from pathlib import Path

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from customer_support_agent.config.settings import CsaEvalSettings, get_agent_settings
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS
from customer_support_agent.persistence.repositories import RefundLedgerRepository
from customer_support_agent.persistence.schema import ensure_schema

_POLICY_SEED = Path(__file__).resolve().parents[1] / "dummy_data" / "agent_policy_seed.sql"
_AGENT_ID_TOKEN = "__AGENT_ID__"
_AGENT_VERSION_TOKEN = "__AGENT_VERSION__"
_ORDER_KEY = "order_id"


class FixtureOrder(BaseModel):
    order_id: int
    user_id: int
    order_status: str
    refund_status: str
    amount_cents: int


class FixtureUser(BaseModel):
    user_id: int
    user_code: str
    name: str
    email: str
    phone: str | None = None


class FixtureFacts(BaseModel):
    """What the fixture says is true — the oracle's ground truth."""

    users: dict[str, FixtureUser]
    orders: dict[int, FixtureOrder]
    direct_limit_cents: int
    seeded_refund_ids: list[str]
    seeded_review_ids: list[int]
    # Orders a review was already waiting on before the case began.
    seeded_review_orders: list[int]

    def user(self, code: str) -> FixtureUser:
        return self.users[code]


class EvalFixture:
    """Builds the template database and hands out per-case copies."""

    def __init__(self, settings: CsaEvalSettings | None = None) -> None:
        self._settings = settings or get_agent_settings().evals
        self._template = self._settings.resolved_fixture_db_path
        self._workspace = self._settings.resolved_workspace_dir

    @property
    def template_path(self) -> Path:
        return self._template

    def build(self) -> Path:
        """(Re)build the template from schema, seed and policy. Idempotent."""
        self._template.parent.mkdir(parents=True, exist_ok=True)
        self._template.unlink(missing_ok=True)
        pool = SqliteConnectionPool(self._template)
        try:
            ensure_schema(pool)
            RefundLedgerRepository(pool)
            pool.execute_script(self._settings.resolved_fixture_seed_path.read_text())
            pool.execute_script(policy_seed_sql())
        finally:
            pool.close()
        return self._template

    def ensure_built(self) -> Path:
        """Build the template unless a current one exists."""
        if not self._template.exists() or self._stale():
            return self.build()
        return self._template

    def copy_for(self, run_key: str, case_id: str) -> Path:
        """A private copy of the template for one case."""
        target = self._workspace / run_key / f"{case_id}.db"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.ensure_built(), target)
        return target

    def discard(self, path: Path) -> None:
        """Remove a case copy unless configuration says to keep it."""
        if not self._settings.keep_case_workspaces:
            path.unlink(missing_ok=True)

    def facts(self) -> FixtureFacts:
        """Ground truth, read from the template with plain SQL."""
        with sqlite3.connect(self.ensure_built()) as connection:
            connection.row_factory = sqlite3.Row
            users = [
                FixtureUser(**dict(row))
                for row in connection.execute(
                    "SELECT user_id, user_code, name, email, phone FROM users"
                )
            ]
            orders = [
                FixtureOrder(**dict(row))
                for row in connection.execute(
                    "SELECT order_id, user_id, order_status, refund_status, amount_cents FROM orders"
                )
            ]
            limit = connection.execute(
                "SELECT per_action_limit_cents FROM platform_agent_budgets WHERE action_class = ?",
                (REFUND_ACTION_CLASS,),
            ).fetchone()
            refund_ids = [row[0] for row in connection.execute("SELECT refund_id FROM refunds")]
            reviews = list(connection.execute("SELECT id, detail FROM platform_review_queue"))
        return FixtureFacts(
            users={user.user_code: user for user in users},
            orders={order.order_id: order for order in orders},
            direct_limit_cents=int(limit[0]) if limit else 0,
            seeded_refund_ids=refund_ids,
            seeded_review_ids=[row[0] for row in reviews],
            seeded_review_orders=[_review_order(row[1]) for row in reviews],
        )

    def _stale(self) -> bool:
        built = self._template.stat().st_mtime
        sources = (self._settings.resolved_fixture_seed_path, _POLICY_SEED)
        return any(source.stat().st_mtime > built for source in sources)


def _review_order(detail: str) -> int:
    """The order a seeded review item is about (its `detail` JSON)."""
    return int(json.loads(detail)[_ORDER_KEY])


def policy_seed_sql() -> str:
    """This agent's runtime policy (limits, permissions), for its configured identity."""
    identity = get_settings().agent
    return (
        _POLICY_SEED.read_text()
        .replace(_AGENT_ID_TOKEN, identity.id)
        .replace(_AGENT_VERSION_TOKEN, identity.version)
    )
