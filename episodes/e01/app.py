"""E01 — Agent = Model + Harness.

The cold open is one question asked twice. A customer asks for a $4,000
refund; the agent proposes the refund tool; a single authority check
refuses it. Then `/harness off` removes that check and the identical
question, put to the identical model with the identical prompt, moves
$4,000.

That is the whole series in one screen. The model did not get better or
worse between the two answers — nothing about it changed at all. What
changed was whether anything stood between what it proposed and what
was executed.

The episode's only real work is arranging for there to be something
worth refusing: the demo data's largest order is under $500, and an
unsupervised $500 refund does not frighten anybody.
"""

from agentic_commons.config import YamlBackedSettings
from agentic_commons.logging import get_logger
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import Order
from customer_support_agent.persistence.repositories import OrdersRepository, UsersRepository
from customer_support_agent.persistence.schema import ensure_schema


class RefundScenario(YamlBackedSettings):
    """The order this episode's cold open turns on."""

    _yaml_section = "episode.scenario"

    order_id: int
    amount: float
    order_status: str


def setup() -> None:
    """Put one large, settled order in front of the demo customer.

    Written every run rather than only when missing: the episode ends
    with the order refunded, and the next take needs it refundable
    again. Idempotent by order id, so it does not accumulate.
    """
    scenario = RefundScenario()  # type: ignore[call-arg]
    pool = SqliteConnectionPool()
    ensure_schema(pool)
    customer = UsersRepository(pool).get_by_code(get_agent_settings().demo_user_code)
    if customer is None:
        # No demo customer means an unseeded database. Say so rather than
        # failing later inside a tool call, where it reads as a bug in
        # the agent rather than as missing data.
        get_logger("episodes").warning(
            "Demo customer not found; run `make dummy-data` before this episode"
        )
        return
    OrdersRepository(pool).save(
        Order(
            order_id=scenario.order_id,
            user_id=customer.user_id,
            timestamp=utc_now().isoformat(),
            order_status=scenario.order_status,  # type: ignore[arg-type]
            refund_status="none",
            amount=scenario.amount,
        )
    )
