"""The evidence the customer support agent hands the guard, against its real prompts and corpus.

`test_grounding_cascade.py` writes its evidence by hand. This one builds it
the way `finalize` does — the real policy file, the real knowledge
articles, the real system and reply prompts — so a wording change to any
of them that makes a correct reply unverifiable fails here.
"""

from types import SimpleNamespace
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.evals import create_scoring_cascade
from agentic_commons.evals.entailment import EntailmentResult
from agentic_commons.guardrails.response import ResponseGuardrail
from agentic_commons.hallucination_protection import NOT_EXECUTED_MARKER, DetectionLadder
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState

_ORDER = 10864


class _Support(NodeSupport):
    def __init__(self) -> None:
        self._agent_policy: Any = SimpleNamespace(per_action_limit=lambda _action: None)
        self._registry: Any = SimpleNamespace(
            get=lambda name: SimpleNamespace(side_effecting=name.endswith("process_refund"))
        )


def _state(*actions: tuple[str, ...], **metadata: Any) -> AgentState:
    state = AgentState(query="Refund my order", metadata=metadata)
    for name, result, *outcome in actions:
        state.working_memory.add_current_turn(
            {
                "kind": "action",
                "name": name,
                "arguments": {},
                "result": result,
                "outcome": outcome[0] if outcome else "completed",
            }
        )
    return state


def _guard() -> ResponseGuardrail:
    return ResponseGuardrail(DetectionLadder(create_scoring_cascade()))


class TestTheReplyPromptIsEvidence:
    def test_the_manner_the_reply_was_asked_for_is_a_rule_it_may_restate(self) -> None:
        rules = [c.content for c in _Support()._policy_evidence() if c.kind == "system"]
        assert any("Address them directly" in rule for rule in rules)


class TestAnEffectIsEvidenceOnlyOnceItIsReadBack:
    """The deterministic tier: the system of record decides what a tool result is worth.

    `EffectVerifier` re-reads the record after a write. A call whose
    effect did not take is recorded as *unverified*, and grounding then
    sees it as a call that did not execute rather than as a result to
    believe. Nothing about it needs to be phrased as evidence.
    """

    def test_an_unverified_call_is_admitted_as_not_executed(self) -> None:
        state = _state(
            ("process_refund", f"Refund of $18.70 for order {_ORDER} completed", "unverified")
        )
        [chunk] = _Support()._grounding_evidence(state)[:1]
        assert chunk.kind == "tool_result"
        assert NOT_EXECUTED_MARKER in chunk.content

    async def test_a_refund_claim_against_an_unverified_call_is_refuted(self) -> None:
        state = _state(
            (
                "process_refund",
                f"The refund of $18.70 for order {_ORDER} has been completed",
                "unverified",
            )
        )
        result = await _guard().enforce(
            f"Order {_ORDER} has been refunded.", _Support()._grounding_evidence(state)
        )
        assert result.escalated

    def test_basic_sanity_is_a_gate_not_evidence(self) -> None:
        """Pre-flight prose about the state *before* the action grounds no outcome."""
        sanity = {
            "findings": [
                {
                    "check": "ownership",
                    "passed": True,
                    "reason": "Order exists and belongs to the authenticated customer.",
                }
            ]
        }
        evidence = _Support()._grounding_evidence(_state(basic_sanity=sanity))
        assert not any("authenticated customer" in chunk.content for chunk in evidence)


class TestALoopStepIsNamedForWhatItIs:
    def test_the_step_label_does_not_claim_react(self) -> None:
        support = _Support()
        support._loop_step, support._max_rounds = 3, 5
        _, fields = support._stepped("llm_request", "x", None)
        assert fields is not None
        assert fields["loop_step"] == "3 of 5 (model call → tool → result)"

    def test_the_wording_is_configuration(self) -> None:
        settings = get_settings().observability
        assert "observe" not in settings.loop_step_detail
        assert "{step}" in settings.loop_step_prefix


class TestTheReportedRepliesAgainstTheRealCorpus:
    async def _enforce(self, reply: str, *actions: tuple[str, ...]) -> Any:
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

    async def test_the_reply_from_the_live_run_keeps_both_sentences(self) -> None:
        """A BasicSanity line once "contradicted" the second sentence at 0.99."""
        sanity = {
            "findings": [
                {
                    "check": "ownership",
                    "passed": True,
                    "reason": "Order exists and belongs to the authenticated customer.",
                }
            ]
        }
        support = _Support()
        state = _state(
            (
                "process_refund",
                "The refund of $32.94 for order 10829 has been completed. Order 10829 refund status: completed",
            ),
            basic_sanity=sanity,
        )
        reply = "Your refund for order 10829 has been completed. $32.94 has been refunded."
        result = await _guard().enforce(reply, support._grounding_evidence(state))
        assert not result.escalated, result.claim_checks
        assert result.response == reply


class TestTheCardExplainsAnUnverifiedClaim:
    def test_it_prints_the_reason_not_only_the_closest_passage(self) -> None:
        from rich.text import Text

        from agentic_commons.tui.widgets import _append_claim

        body = Text()
        _append_claim(
            body,
            {
                "claim": "Refunds take a while to arrive.",
                "verdict": "undetermined",
                "rationale": "policy: nothing related (best 0.21 < 0.3) · system: no evidence",
                "evidence": [
                    {"tier": "policy", "nli_label": "neutral", "nli_score": 0.98, "excerpt": "x"}
                ],
            },
        )
        assert "nothing related (best 0.21 < 0.3)" in body.plain


class TestTheRefusalFromTheLiveRun:
    async def test_the_reply_is_supported_by_the_call_that_was_refused(self) -> None:
        state = AgentState(query="want refund for this order -> 10786")
        state.working_memory.add_current_turn(
            {
                "kind": "action",
                "name": "support__process_refund",
                "arguments": {
                    "reason": "Customer requested refund",
                    "order_id": 10786,
                    "amount": "90.93",
                },
                "result": "'process_refund' names order:10786, which is not a record this "
                "conversation established for $90.93.",
                "outcome": "blocked",
            }
        )
        reply = "I’m unable to process a refund for order 10786 at this time."
        result = await _guard().enforce(reply, _Support()._grounding_evidence(state))
        assert not result.escalated
        assert result.supported == 1
        assert result.response == reply


class TestTheTurnWhereARefundWasReferredAfterAStatusRead:
    def _state(self) -> AgentState:
        return _state(
            (
                "support__get_refund_status",
                "No refund has been requested for order 10169. Order 10169 refund status: none",
            ),
            (
                "support__process_refund",
                "Refund of $73.63 for order 10169 has been queued for human review",
                "referred",
            ),
        )

    def test_only_the_write_is_marked_state_changing(self) -> None:
        tools = [
            c for c in _Support()._grounding_evidence(self._state()) if c.kind == "tool_result"
        ]
        flags = {c.content: c.metadata["state_changing"] for c in tools}
        assert flags["No refund has been requested for order 10169."] is False
        assert flags["Refund of $73.63 for order 10169 has been queued for human review"] is True

    async def test_the_reply_is_not_blocked_by_the_read_that_came_first(self) -> None:
        reply = (
            "Your refund request for order 10169 has been received and is currently pending review."
        )
        result = await _guard().enforce(reply, _Support()._grounding_evidence(self._state()))
        assert not result.escalated, result.claim_checks
        assert result.response == reply


class _Scripted:
    """The hosted model on order 10169: sure the stale read contradicts, neutral on the rest."""

    model_name = "stub-nli"

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        stale = premise.startswith("No refund has been requested")
        return EntailmentResult(
            label="contradiction" if stale else "unknown",
            entailed=False,
            score=1.0 if stale else 0.99,
            calibrated=True,
            model="stub-nli",
        )


class TestTheHostedModelOnThatTurn:
    async def test_the_reply_survives_when_the_later_observation_is_only_neutral(self) -> None:
        state = _state(
            (
                "support__get_refund_status",
                "No refund has been requested for order 10169. Order 10169 refund status: none",
            ),
            (
                "support__process_refund",
                "Refund of $73.63 for order 10169 has been queued for human review",
                "referred",
            ),
        )
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade(), entailment=_Scripted()))
        reply = (
            "Your refund request for order 10169 has been received and is currently pending review."
        )
        result = await guard.enforce(reply, _Support()._grounding_evidence(state))
        assert not result.escalated, result.claim_checks


class _NeverReadsARecord:
    """The hosted model on order 10665: it contradicts any state record, and must never see one."""

    model_name = "stub-nli"

    def __init__(self) -> None:
        self.premises: list[str] = []

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        self.premises.append(premise)
        stale = "refund status:" in premise
        return EntailmentResult(
            label="contradiction" if stale else "unknown",
            entailed=False,
            score=0.99,
            calibrated=True,
            model="stub-nli",
        )


class TestAStateRecordIsNotSentToNli:
    """`Order 10665 refund status: pending_review.` "contradicted" `has been submitted` at 0.99."""

    def _turn(self) -> AgentState:
        return _state(
            (
                "support__process_refund",
                "Refund of $73.63 for order 10665 has been queued for human review. "
                "Order 10665 refund status: pending_review.",
                "referred",
            )
        )

    async def test_both_replies_from_the_live_run_are_supported(self) -> None:
        nli = _NeverReadsARecord()
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade(), entailment=nli))
        reply = "Your refund request for order 10665 has been submitted and is now pending review."
        result = await guard.enforce(reply, _Support()._grounding_evidence(self._turn()))
        assert not result.escalated, result.claim_checks
        assert result.supported == result.claims
        assert not any("refund status:" in premise for premise in nli.premises)

    async def test_the_card_quotes_the_line_that_decided_it(self) -> None:
        from rich.text import Text

        from agentic_commons.tui.widgets import _append_claim

        guard = ResponseGuardrail(
            DetectionLadder(create_scoring_cascade(), entailment=_NeverReadsARecord())
        )
        result = await guard.enforce(
            "Your refund request for order 10665 is now pending review.",
            _Support()._grounding_evidence(self._turn()),
        )
        body = Text()
        _append_claim(body, result.claim_checks[0])
        assert (
            'Supported By [Tool Result] (Record): "Order 10665 refund status: pending_review."'
            in body.plain
        )
