"""The order, the policy and the session as typed facts, for the hallucination check.

The guard reads a reply's claims against these values before any model does
(`agent_platform.hallucination_check.ledger`): a delivery date, a total, whether
a refund is possible, how long one takes to settle, who the customer is. Each fact
also carries a sentence of its own, so NLI still has a self-contained premise and a
corrected reply has the record's words to put back.

Which attributes exist and how a claim words them is configuration
(`hallucination_check.facts` in this agent's config); the sentences are
`customer_support_agent.grounding_facts.statements`. The values come from the reads
the turn already made: the order row, the settings, the authenticated customer.
"""

from typing import Literal

from agent_platform.context.types import Chunk
from agent_platform.core.primitives.money import Money
from agent_platform.hallucination_check.ledger import FLAG_TRUE, Fact, fact_chunk
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.identity import Customer
from customer_support_agent.domain.models import Order
from customer_support_agent.presentation.orders import refund_state_sentence

FactKind = Literal["tool_result", "policy", "memory"]

# The evidence tier each source's facts belong to, and the source a card names.
_RECORD: FactKind = "tool_result"
_POLICY: FactKind = "policy"
_SESSION: FactKind = "memory"
_ORDER_SOURCE = "order_record:{order_id}"
_POLICY_SOURCE = "policy:{attribute}"
_SESSION_SOURCE = "session:{attribute}"
_FLAG_FALSE = "false"
# How many characters of an ISO timestamp are its date.
_DATE_CHARS = 10
_SECONDS_PER_MINUTE = 60


def order_facts(order: Order) -> list[Chunk]:
    """What the order row states: status, dates, total, product, refund and eligibility."""
    subject = str(order.order_id)
    stated = _statements()
    values = {
        "order_status": order.order_status,
        "product": order.description,
        "order_total": _amount(order.amount),
        "delivered_on": (order.delivered_at or "")[:_DATE_CHARS],
        "due_on": (order.estimated_delivery_date or "")[:_DATE_CHARS],
        # Where it is: the carrier's own words, which a reply may repeat exactly.
        "carrier": order.carrier or "",
        "tracking_number": order.tracking_number or "",
        "current_location": order.current_location or "",
    }
    facts = [
        _fact(subject, name, value, stated[name].format(order_id=subject, value=value), _RECORD)
        for name, value in values.items()
        if value
    ]
    return [
        *(fact_chunk(fact) for fact in facts),
        fact_chunk(_refund_status(order, subject)),
        fact_chunk(_eligibility(order, subject, stated)),
    ]


def policy_facts(refund_limit: Money | None) -> list[Chunk]:
    """What the rules state: how long a refund takes to settle, the limit, who to contact."""
    settings = get_agent_settings()
    stated = _statements()
    window = settings.settlement.window_seconds
    facts = [
        _fact(
            "",
            "settlement_window",
            str(window),
            stated["settlement_window"].format(minutes=window // _SECONDS_PER_MINUTE),
            _POLICY,
        ),
        _fact(
            "",
            "support_contact",
            settings.support_contact,
            stated["support_contact"].format(value=settings.support_contact),
            _POLICY,
        ),
    ]
    if refund_limit is not None:
        value = _amount(refund_limit)
        facts.append(
            _fact("", "refund_limit", value, stated["refund_limit"].format(value=value), _POLICY)
        )
    return [fact_chunk(fact) for fact in facts]


def session_facts(customer: Customer | None) -> list[Chunk]:
    """Who the authenticated customer is: established by authentication, never by the customer."""
    if customer is None:
        return []
    stated = _statements()
    values = {
        "customer_name": customer.name,
        "customer_id": str(customer.customer_id),
        "verified": FLAG_TRUE if customer.verified else _FLAG_FALSE,
    }
    return [
        fact_chunk(_fact("", name, value, stated[name].format(value=value), _SESSION))
        for name, value in values.items()
        if value and (name != "verified" or customer.verified)
    ]


def _refund_status(order: Order, subject: str) -> Fact:
    """The refund's state, in the sentence the refund tool states it with."""
    return _fact(
        subject, "refund_status", order.refund_status, refund_state_sentence(order), _RECORD
    )


def _eligibility(order: Order, subject: str, stated: dict[str, str]) -> Fact:
    """Whether a refund is possible now: why not, in the record's words, when it is not."""
    reason = order.refund_ineligibility_reason
    if order.refund_declined:
        settings = get_agent_settings()
        escalation = settings.refund_declined_escalation.format(contact=settings.support_contact)
        reason = f"{reason}. {escalation}"
    statement = reason or stated["refund_eligible"].format(order_id=subject)
    value = _FLAG_FALSE if reason else FLAG_TRUE
    return _fact(subject, "refund_eligible", value, statement, _RECORD)


def _fact(subject: str, attribute: str, value: str, statement: str, kind: FactKind) -> Fact:
    sources = {_RECORD: _ORDER_SOURCE, _POLICY: _POLICY_SOURCE, _SESSION: _SESSION_SOURCE}
    source = sources[kind].format(order_id=subject, attribute=attribute)
    return Fact(
        subject=subject,
        attribute=attribute,
        value=value,
        statement=statement,
        kind=kind,
        source=source,
    )


def _amount(amount: Money) -> str:
    """An amount as a sentence writes it, and as the ledger parses it back."""
    return str(amount)


def _statements() -> dict[str, str]:
    return get_agent_settings().grounding_facts.statements
