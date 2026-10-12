"""Which order the customer means, and the mandate that follows from it.

The live failure: order 10781 had been discussed, and the customer wrote
"want refund for this order -> 10786". Neither extractor read that as
naming an order, so the old one was carried, the mandate covered 10781,
and the Gatekeeper refused a refund on the order the customer had just
named. Every case here is a way of saying it.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.memory.working import WorkingMemory
from agent_platform.primitives.money import Money
from customer_support_agent.config import get_agent_settings
from customer_support_agent.deterministic_checks.order_validator import OrderChecks
from customer_support_agent.domain.order_reference import (
    ORDER_UNDER_DISCUSSION,
    OrderReferenceResolver,
)
from customer_support_agent.gatekeeper.mandates import EstablishedOrderScope
from customer_support_agent.intent.phrases import PhraseReader

_CUSTOMER = 7
_RULES = get_agent_settings().order_reference
_OWNED = {10781, 10786}


def _owns(order_id: int, customer_id: int) -> bool:
    return customer_id == _CUSTOMER and order_id in _OWNED


def _memory(order_id: int | None = 10781) -> WorkingMemory:
    memory = WorkingMemory()
    if order_id is not None:
        memory.set(ORDER_UNDER_DISCUSSION, f"order {order_id}")
    return memory


def _resolve(query: str, *, owns: Any = _owns, customer: int | None = _CUSTOMER) -> int | None:
    return OrderReferenceResolver(_RULES, owns=owns).resolve(
        query, {}, _memory(), customer_id=customer
    )


class TestTheCustomerNamesAnOrderInTheirOwnWords:
    @pytest.mark.parametrize(
        "query",
        [
            "want refund for this order -> 10786",
            "order: 10786",
            "order no. 10786",
            "order number 10786 please",
            "the order is 10786",
            "this order, 10786",
            "refund 10786",
            "cancel order #10786",
            "10786 order",
        ],
    )
    def test_the_named_order_beats_the_carried_one(self, query: str) -> None:
        assert _resolve(query) == 10786

    def test_a_bare_number_that_is_their_own_order_counts(self) -> None:
        assert _resolve("10786 please") == 10786

    def test_an_order_named_but_not_theirs_still_resolves_so_it_can_be_refused(self) -> None:
        """DeterministicCheck says "not on your account"; it cannot if nothing is named."""
        assert _resolve("order 99999") == 99999


class TestAMessageNeverFallsBackToTheOldOrderBehindTheCustomersBack:
    def test_a_number_that_is_not_theirs_resolves_to_nothing(self) -> None:
        assert _resolve("my zip is 90210, refund it") is None

    def test_two_of_their_orders_without_an_anchor_is_ambiguous(self) -> None:
        assert _resolve("10781 or 10786?") is None

    def test_no_number_carries_the_order_from_earlier(self) -> None:
        assert _resolve("yes, refund it") == 10781

    def test_a_bare_number_is_not_read_without_a_customer_to_check_it_against(self) -> None:
        assert _resolve("10786 please", customer=None) == 10781

    def test_a_bare_number_is_not_read_without_an_ownership_check(self) -> None:
        assert _resolve("10786 please", owns=None) == 10781


class TestThePathThatMovesMoney:
    def test_the_live_sequence_yields_a_mandate_for_the_named_order(self) -> None:
        resolved = _resolve("want refund for this order -> 10786")
        orders = SimpleNamespace(
            get_for_customer=lambda order_id, customer_id: (
                SimpleNamespace(amount=Money.parse("90.93"))
                if _owns(order_id, customer_id)
                else None
            )
        )
        mandates = EstablishedOrderScope(orders).mandates_for(  # type: ignore[arg-type]
            str(_CUSTOMER), "", 2, {ORDER_UNDER_DISCUSSION: resolved}
        )
        assert {m.target for m in mandates} == {"order:10786"}
        assert {m.max_amount for m in mandates} == {Money.parse("90.93")}


class TestDeterministicChecksReadTheSameReference:
    """Two extractors that disagree is how an agent answers about one order and acts on another."""

    @pytest.mark.parametrize(
        "query",
        [
            "want refund for this order -> 10786",
            "order: 10786",
            "refund 10786",
            "10786 please",
            "my zip is 90210, refund it",
            "yes, refund it",
        ],
    )
    def test_it_agrees_with_the_resolver_on_what_the_message_names(self, query: str) -> None:
        resolver = OrderReferenceResolver(_RULES, owns=_owns)
        assert self._checks(resolver).order_id(query, {}, _CUSTOMER) == resolver.resolve(
            query, {}, None, customer_id=_CUSTOMER
        )

    def test_it_reads_the_reference_the_turn_already_settled(self) -> None:
        checks = self._checks(OrderReferenceResolver(_RULES, owns=_owns))
        assert (
            checks.order_id("yes, refund it", {ORDER_UNDER_DISCUSSION: 10781}, _CUSTOMER) == 10781
        )

    def test_the_gatekeepers_final_recheck_still_reads_the_validated_call(self) -> None:
        checks = self._checks(OrderReferenceResolver(_RULES, owns=_owns))
        assert checks.order_id("", {"tool_params": {"order_id": 10786}}, _CUSTOMER) == 10786

    @staticmethod
    def _checks(resolver: OrderReferenceResolver) -> OrderChecks:
        return OrderChecks(
            SimpleNamespace(),  # type: ignore[arg-type]
            refund_policy=SimpleNamespace(),  # type: ignore[arg-type]
            cancellation_policy=SimpleNamespace(),  # type: ignore[arg-type]
            references=resolver,
        )


class TestTheCustomerMovesFromOneOrderToAnother:
    """ "do not refund 10781, but 10786": the refused order is named, the bare one is meant."""

    @staticmethod
    def _refusing(query: str) -> int | None:
        resolver = OrderReferenceResolver(_RULES, owns=_owns, negated_at=PhraseReader().negated_at)
        return resolver.resolve(query, {}, _memory(None), customer_id=_CUSTOMER)

    def test_the_order_they_moved_to_is_the_one_under_discussion(self) -> None:
        assert self._refusing("do not refund order 10781, but 10786.") == 10786

    def test_a_refusal_alone_names_no_order(self) -> None:
        assert self._refusing("do not refund order 10781") is None

    def test_the_refused_order_is_never_chosen(self) -> None:
        assert self._refusing("do not refund order 10781, but 10781") is None
