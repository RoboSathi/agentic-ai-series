"""The support agent's reported replies, re-checked with its typed facts in evidence.

Each reply below was judged wrong in a live run while the record that decides it was
in hand. The order, the rules and the session now travel as typed facts
(`presentation/facts.py`), which the guard reads before any model: the delivery
date, eligibility, the settlement window, the verified customer, the limit to compare
a total against. The model is never asked (tests are offline), so what decides each
claim here is the record itself.
"""

from agent_platform.context.types import Chunk
from agent_platform.guardrails import ResponseGuardrail, ResponseGuardrailResult
from agent_platform.hallucination_check import DetectionLadder
from agent_platform.hallucination_check.direct import RECORD_STATE_MISMATCH
from agent_platform.primitives.money import Money
from agent_platform.scoring import create_scoring_cascade
from customer_support_agent.domain.identity import Customer
from customer_support_agent.domain.models import Order
from customer_support_agent.presentation.facts import order_facts, policy_facts, session_facts
from customer_support_agent.presentation.orders import REFUND_STATUS_LINE

_LIMIT = Money.parse("50.00")


def _order(order_id: int, refund_status: str = "none", amount: str = "67.27") -> Order:
    return Order.model_validate(
        {
            "order_id": order_id,
            "user_id": 2,
            "timestamp": "2026-07-30T10:00:00+00:00",
            "order_status": "delivered",
            "refund_status": refund_status,
            "amount": amount,
            "product_name": "Laptop Stand, Aluminium",
            "delivered_at": "2026-08-02T10:00:00+00:00",
        }
    )


def _evidence(order: Order) -> list[Chunk]:
    status = REFUND_STATUS_LINE.format(order_id=order.order_id, status=order.refund_status)
    return [
        Chunk(content=status, kind="tool_result", source="order_record"),
        *order_facts(order),
        *policy_facts(_LIMIT),
        *session_facts(Customer(customer_id=2, name="Dennis Johnson")),
    ]


async def _checked(reply: str, order: Order) -> ResponseGuardrailResult:
    guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
    return await guard.enforce(reply, _evidence(order))


def _verdicts(result: ResponseGuardrailResult) -> dict[str, str]:
    return {check["claim"]: check["verdict"] for check in result.claim_checks}


class TestWhatTheRecordStatesIsSupported:
    async def test_who_the_customer_is(self) -> None:
        """Reported UNVERIFIED: the session's identity was a sentence NLI read as neutral."""
        result = await _checked(
            "You're Dennis Johnson, verified customer with account #2.", _order(10986)
        )
        assert set(_verdicts(result).values()) == {"supported"}
        assert not result.escalated

    async def test_when_an_issued_refund_settles(self) -> None:
        """Reported UNVERIFIED: the 60-minute window was a setting no evidence stated."""
        order = _order(10245, refund_status="issued", amount="20.31")
        result = await _checked("It will settle within 60 minutes.", order)
        assert set(_verdicts(result).values()) == {"supported"}

    async def test_delivery_eligibility_and_the_limit_comparison(self) -> None:
        """Reported: eligibility CONTRADICTED by 'no refund requested'; the comparison UNVERIFIED."""
        reply = (
            "Your order 10986 was delivered on 2026-08-02 and is eligible for a refund. "
            "However, the refund amount of $67.27 exceeds the $50.00 limit I can handle directly."
        )
        result = await _checked(reply, _order(10986))
        verdicts = _verdicts(result)
        assert set(verdicts.values()) == {"supported"}, verdicts
        assert not result.escalated


class TestAWrongStatusIsCorrectedFromTheRecord:
    async def test_the_cut_sentence_is_replaced_by_the_records_own(self) -> None:
        """A refund still settling, said to be complete: the record's sentence goes instead."""
        order = _order(10226, refund_status="issued")
        result = await _checked("Your refund for order 10226 has been completed.", order)
        assert result.reason_code == RECORD_STATE_MISMATCH
        assert not result.withheld
        assert result.response.startswith(
            "The refund of $67.27 for order 10226 has been issued and is waiting to settle"
        )
        assert result.corrected_sentences


class TestAMadeUpClaimIsCut:
    async def test_a_claim_nothing_grounds_is_removed_and_queued(self) -> None:
        reply = "Your order 10986 was delivered on 2026-08-02. A reviewer will call you tonight."
        result = await _checked(reply, _order(10986))
        assert result.escalated and not result.withheld
        assert result.removed_sentences == ["A reviewer will call you tonight."]
        assert "unsupported" in _verdicts(result).values()


class TestAFactAboutThisCaseIsNeverCheckedAgainstPolicy:
    async def test_where_the_refund_stands_is_read_from_the_record(self) -> None:
        """Reported: "no refund has been requested yet" CONTRADICTED by a KB FAQ line.

        The clause names no order, so it was classed as a rule and read against the
        knowledge base. It states where this refund stands: the record decides it.
        """
        faq = Chunk(
            content=(
                "Frequently asked questions > Can I get a refund for an order that has not "
                "arrived yet? No. A refund applies once an order is delivered."
            ),
            kind="policy",
            source="kb:faq.md#2",
        )
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        order = _order(10105)
        result = await guard.enforce(
            "Your order 10105 has been delivered and no refund has been requested yet.",
            [*_evidence(order), faq],
        )
        verdicts = _verdicts(result)
        assert verdicts["no refund has been requested yet"] == "supported", verdicts
        assert not result.escalated


class TestADeclinedRefundIsFinal:
    async def test_saying_it_cannot_be_refunded_again_is_true(self) -> None:
        """Reported: "declined, so it cannot be refunded again" was CONTRADICTED by eligibility."""
        order = _order(10125, refund_status="denied")
        result = await _checked(
            "Your refund request for order 10125 was already declined, so it cannot be "
            "refunded again.",
            order,
        )
        verdicts = _verdicts(result)
        assert set(verdicts.values()) == {"supported"}, verdicts
        assert not result.escalated

    def test_the_policy_refuses_a_second_request(self) -> None:
        from customer_support_agent.domain.refund_policy import REFUND_DECLINED_FINAL, RefundPolicy

        order = _order(10125, refund_status="denied")
        lookup = type("_One", (), {"get_for_customer": lambda self, *_: order})()
        result = RefundPolicy(lookup).validate(10125, 2, order.amount, action="process_refund")
        assert (result.allowed, result.reason_code) == (False, REFUND_DECLINED_FINAL)
