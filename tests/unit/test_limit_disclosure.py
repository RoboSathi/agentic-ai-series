"""A reply never tells the customer the refund limit, whatever the model wrote."""

from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
from customer_support_agent.config import get_agent_settings
from customer_support_agent.evals.fixture import policy_seed_sql
from customer_support_agent.graph import build_agent
from customer_support_agent.security.limit_disclosure import LimitDisclosureCheck
from tests.conftest import ScriptedLLM

_LIMIT = Money.from_cents(5000)
_STATED = "Order 10622 is $76.66, which exceeds my $50.00 limit, so it needs review."
_RECORD = ["Order 10622 placed 2026-09-01 has shipped and was delivered, $76.66, refund none"]


def test_a_reply_stating_the_limit_is_replaced_by_the_decline() -> None:
    settings = get_agent_settings()
    sent = LimitDisclosureCheck(_LIMIT).withhold(_STATED, _RECORD, 10622)
    assert sent == settings.refund_limit_withheld.with_order.format(
        order_id=10622, contact=settings.support_contact
    )
    assert "$50.00" not in sent


def test_with_no_order_the_decline_names_none() -> None:
    sent = LimitDisclosureCheck(_LIMIT).withhold(_STATED, [], None)
    assert sent is not None and "{order_id}" not in sent and "$50.00" not in sent


def test_an_order_total_equal_to_the_limit_is_quoted_as_read() -> None:
    record = ["Order 10777 placed 2026-09-01 was delivered, $50.00, refund none"]
    reply = "Order 10777's total is $50.00."
    assert LimitDisclosureCheck(_LIMIT).withhold(reply, record, 10777) is None


def test_a_reply_without_the_limit_goes_as_written() -> None:
    reply = "The refund for order 10622 needs a person's review."
    assert LimitDisclosureCheck(_LIMIT).withhold(reply, _RECORD, 10622) is None


def test_no_declared_limit_withholds_nothing() -> None:
    assert LimitDisclosureCheck(None).withhold(_STATED, _RECORD, 10622) is None


async def test_a_draft_sent_as_written_is_still_checked(
    seeded_pool: SqliteConnectionPool, restore_settings: None
) -> None:
    """The common path — the model's own draft, no second call — once skipped the check."""
    seeded_pool.execute_script(policy_seed_sql())
    agent = build_agent(
        llm=ScriptedLLM([{"content": "Order 10002 is above my $50.00 limit, so it needs review."}]),
        pool=seeded_pool,
        with_memory=False,
    )
    state = await agent.handle("refund order 10002")
    assert "$50.00" not in state.response
    assert "disclosure: refund limit withheld" in state.decisions
