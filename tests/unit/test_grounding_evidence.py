"""The evidence the customer support agent hands the guard, against its real prompts and corpus.

`test_grounding_cascade.py` writes its evidence by hand. This one builds it
the way `finalize` does — the real policy file, the real knowledge
articles, the real system and reply prompts — so a wording change to any
of them that makes a correct reply unverifiable fails here.
"""

from types import SimpleNamespace
from typing import Any

from agentic_commons.evals import create_scoring_cascade
from agentic_commons.guardrails.response import ResponseGuardrail
from agentic_commons.hallucination_protection import DetectionLadder
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState

_ORDER = 10864


class _Support(NodeSupport):
    def __init__(self) -> None:
        self._agent_policy: Any = SimpleNamespace(per_action_limit=lambda _action: None)


def _state(*actions: tuple[str, str], **metadata: Any) -> AgentState:
    state = AgentState(query="Refund my order", metadata=metadata)
    for name, result in actions:
        state.working_memory.add_current_turn(
            {
                "kind": "action",
                "name": name,
                "arguments": {},
                "result": result,
                "outcome": "completed",
            }
        )
    return state


def _guard() -> ResponseGuardrail:
    return ResponseGuardrail(DetectionLadder(create_scoring_cascade()))


class TestTheReplyPromptIsEvidence:
    def test_the_manner_the_reply_was_asked_for_is_a_rule_it_may_restate(self) -> None:
        rules = [c.content for c in _Support()._policy_evidence() if c.kind == "system"]
        assert any("Address them directly" in rule for rule in rules)


class TestTheSecondTier:
    def test_a_read_back_is_evidence_and_says_whether_the_effect_took(self) -> None:
        support, state = _Support(), _state()
        support._add_verification(
            state,
            {
                "verification": f"process_refund on order {_ORDER}: refund_status is none",
                "verified": False,
            },
        )
        [chunk] = support._verification_evidence(state)
        assert chunk.kind == "verification"
        assert "NOT verified" in chunk.content

    def test_a_call_with_nothing_to_read_back_adds_nothing(self) -> None:
        support, state = _Support(), _state()
        support._add_verification(state, {"tool_status": "completed"})
        assert support._verification_evidence(state) == []

    def test_what_basic_sanity_established_is_evidence(self) -> None:
        sanity = {
            "findings": [
                {
                    "check": "ownership",
                    "passed": True,
                    "reason": f"Order {_ORDER} belongs to this customer",
                },
                {
                    "check": "shipped",
                    "passed": True,
                    "applicable": False,
                    "reason": "not evaluated",
                },
            ]
        }
        [chunk] = _Support()._verification_evidence(_state(basic_sanity=sanity))
        assert chunk.content == f"Order {_ORDER} belongs to this customer"


class TestTheReportedRepliesAgainstTheRealCorpus:
    async def _enforce(self, reply: str, *actions: tuple[str, str]) -> Any:
        support, state = _Support(), _state(*actions)
        return await _guard().enforce(reply, support._grounding_evidence(state))

    async def test_a_completed_refund_and_where_it_went(self) -> None:
        result = await self._enforce(
            "$18.70 has been refunded to your original payment method.",
            (
                "process_refund",
                f"The refund of $18.70 for order {_ORDER} has been completed. "
                f"Order {_ORDER} refund status: completed",
            ),
        )
        assert not result.escalated, result.claim_checks

    async def test_a_referral_and_what_happens_next(self) -> None:
        result = await self._enforce(
            f"Your refund of $18.70 for order {_ORDER} has been queued for human review "
            "and you will hear back soon.",
            (
                "process_refund",
                f"Refund of $18.70 for order {_ORDER} has been queued for human review",
            ),
        )
        assert not result.escalated, result.claim_checks

    async def test_a_referral_is_not_reported_as_a_completed_refund(self) -> None:
        result = await self._enforce(
            f"$18.70 has been refunded for order {_ORDER}. You will hear back soon.",
            (
                "process_refund",
                f"Refund of $18.70 for order {_ORDER} has been queued for human review",
            ),
        )
        assert result.escalated
        assert "hear back soon" in result.response
