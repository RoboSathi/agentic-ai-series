"""Two live failures: "who am i?" blocked, and `/queue approve 30` finding nothing.

"You are Margaret Bartik, account #12" was contradicted for `12, margaret
appear(s) in the claim but in no evidence`: a name that starts like a month
was read as a date, and the customer's own verified account number was not
evidence. `/queue approve 30` said "no pending request" for a request the
agent had just queued, because the command opened the platform's default
database and the agent had written to its own; and it then told the
customer their refund "was approved" when nothing had been.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.core.primitives.money import Money
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check import DetectionLadder
from agent_platform.hallucination_check.claims import specifics
from agent_platform.hallucination_check.ledger import FLAG_TRUE, facts_in
from agent_platform.scoring import create_scoring_cascade
from customer_support_agent.config import get_agent_settings
from customer_support_agent.console.review_handler import support_queue_command
from customer_support_agent.domain.identity import Customer, set_current_customer
from customer_support_agent.domain.models import HitlRequest
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.nodes.review_wait import _reply
from customer_support_agent.graph.state import AgentState
from customer_support_agent.grounding.evidence import GroundingEvidence
from customer_support_agent.persistence.repositories import HitlQueueRepository
from customer_support_agent.persistence.schema import ensure_schema
from customer_support_agent.review.queue import ReviewOutcome


class TestANameIsNotADate:
    @pytest.mark.parametrize("word", ["Margaret", "Marcus", "Augustine", "Decker", "Junior"])
    def test_a_name_that_starts_like_a_month_is_not_a_specific(self, word: str) -> None:
        assert specifics(f"You are {word} Bartik") == set()

    def test_may_the_verb_is_not_a_date(self) -> None:
        assert specifics("Your refund may take a few days") == set()

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("It arrives on March 3", "3 mar"),
            ("It arrives on Mar 3", "3 mar"),
            ("Shipped 3 Nov", "3 nov"),
            ("Due May 3", "3 may"),
            ("It arrives in March", "march"),
        ],
    )
    def test_a_real_date_still_is(self, text: str, expected: str) -> None:
        assert expected in specifics(text)


class _Support(NodeSupport):
    def __init__(self) -> None:
        self._agent_policy: Any = SimpleNamespace(per_action_limit=lambda _action: None)
        self._registry: Any = SimpleNamespace(get=lambda _name: None)
        self._evidence = GroundingEvidence(self._registry, self._agent_policy)


def _state() -> AgentState:
    return AgentState(query="who am i?", session_id="s", customer_id=12)


def _guard() -> ResponseGuardrail:
    return ResponseGuardrail(DetectionLadder(create_scoring_cascade()))


def _evidence(customer: Customer | None) -> list[Chunk]:
    set_current_customer(customer)
    return _Support()._grounding_evidence(_state())


class TestTheVerifiedCustomerIsEvidence:
    """Who the customer is comes from authentication, as typed facts — never from them."""

    _WHO = Customer(customer_id=12, name="Margaret Bartik")

    def test_the_identity_the_harness_established_is_offered_as_evidence(self) -> None:
        facts = {fact.attribute: (fact, chunk) for fact, chunk in facts_in(_evidence(self._WHO))}
        name, chunk = facts["customer_name"]
        assert (name.value, chunk.kind) == ("Margaret Bartik", "memory")
        assert facts["customer_id"][0].value == "12"
        assert facts["verified"][0].value == FLAG_TRUE

    def test_nothing_is_offered_when_nobody_was_verified(self) -> None:
        sessions = [c for c in _evidence(None) if c.source.startswith("session:")]
        assert sessions == []

    async def test_who_am_i_is_supported(self) -> None:
        result = await _guard().enforce(
            "You are Margaret Bartik, account #12.", _evidence(self._WHO)
        )
        assert not result.escalated, result.claim_checks
        assert result.supported == 1

    async def test_a_different_account_is_still_refused(self) -> None:
        result = await _guard().enforce(
            "You are Margaret Bartik, account #99.", _evidence(self._WHO)
        )
        assert result.escalated

    async def test_nobody_verified_means_no_identity_may_be_stated(self) -> None:
        result = await _guard().enforce("You are Margaret Bartik, account #12.", _evidence(None))
        assert result.escalated


class TestEveryDefaultPoolOpensTheAgentsDatabase:
    """`/queue`, `/memory` and `/audit` open the default pool; the agent writes to its own file."""

    def test_the_platform_setting_and_the_agents_are_one_path(self) -> None:
        assert get_settings().persistence.resolved_db_path == get_agent_settings().resolved_db_path

    def test_a_default_pool_is_the_agents_database(self) -> None:
        assert SqliteConnectionPool().db_path == str(get_agent_settings().resolved_db_path)


class _Runs:
    def waiting_on(self, _key: str) -> Any:
        return SimpleNamespace(thread_id="tui-1:abc")


class _Agent:
    def __init__(self) -> None:
        self.runs = _Runs()
        self.resumed: list[Any] = []

    async def resume(self, thread_id: str, decision: Any = None) -> Any:
        self.resumed.append(decision)
        return SimpleNamespace(status="completed", response="ok")

    async def checkpoint_history(self, thread_id: str) -> list[Any]:
        return []

    def cancel(self) -> bool:
        return False

    async def abort(self, thread_id: str) -> Any:
        return SimpleNamespace(status="aborted", response="")

    def describe_progress(self, values: dict[str, Any]) -> dict[str, Any]:
        return {}


def _service(outcome: ReviewOutcome) -> Any:
    async def approve(request_id: int, reviewer: str) -> ReviewOutcome:
        return outcome

    return SimpleNamespace(approve=approve, deny=lambda request_id, reviewer, reason="": outcome)


async def _decide(
    monkeypatch: pytest.MonkeyPatch, outcome: ReviewOutcome, verb: str = "approve"
) -> Any:
    monkeypatch.setattr(
        "customer_support_agent.graph.builder.build_review_service",
        lambda pool=None, audit=None: _service(outcome),
    )
    agent = _Agent()
    pool = SqliteConnectionPool(":memory:")
    ensure_schema(pool)
    queue = HitlQueueRepository(pool)
    queue.save(HitlRequest(order_id=10003, user_id=2, requested_amount=Money.parse("45.00")))
    command = support_queue_command(pool, agent=lambda: agent)
    result = await command.run([verb, str(queue.last_id()), "reviewed"])
    return agent, result


class TestADecisionThatDecidedNothingDoesNotResumeTheCustomer:
    async def test_a_request_that_is_not_pending_leaves_the_conversation_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, result = await _decide(
            monkeypatch,
            ReviewOutcome(request_id=30, decision="not_found", reason="no pending request"),
        )
        assert agent.resumed == []
        assert any("not pending" in line for line in result.lines)

    async def test_an_approval_resumes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = await _decide(
            monkeypatch,
            ReviewOutcome(request_id=30, decision="approved", order_id=10003, refund_issued=True),
        )
        assert len(agent.resumed) == 1

    async def test_a_denial_resumes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, _ = await _decide(
            monkeypatch, ReviewOutcome(request_id=30, decision="denied", order_id=10003), "deny"
        )
        assert len(agent.resumed) == 1


class TestTheCustomerIsNeverToldOfAnApprovalThatDidNotHappen:
    _PENDING = {"order_id": 10003, "amount": "$219.30"}

    @pytest.mark.parametrize("decision", ["not_found", "something_else", ""])
    def test_an_undecided_request_is_still_under_review(self, decision: str) -> None:
        reply = _reply(self._PENDING, {"decision": decision, "reason": "no pending request"})
        assert "approved" not in reply and "under review" in reply

    def test_a_refund_that_failed_after_approval_says_so(self) -> None:
        reply = _reply(self._PENDING, {"decision": "failed", "reason": "the saga rolled back"})
        assert "approved but could not be completed" in reply
