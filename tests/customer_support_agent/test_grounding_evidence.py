"""The evidence the customer support agent hands the guard, against its real prompts and corpus.

`test_grounding_cascade.py` writes its evidence by hand. This one builds it
the way `finalize` does — the real policy file, the real knowledge
articles, the real system and reply prompts — so a wording change to any
of them that makes a correct reply unverifiable fails here.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check import NOT_EXECUTED_MARKER, DetectionLadder
from agent_platform.hallucination_check.claim_view import claim_check
from agent_platform.hallucination_check.resolver import DIRECT_TIER, NO_TIER
from agent_platform.hallucination_check.types import DetectionVerdict, EvidenceCheck
from agent_platform.scoring import create_scoring_cascade
from agent_platform.scoring.entailment import EntailmentResult
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.grounding.evidence import GroundingEvidence

_ORDER = 10864


class _Support(NodeSupport):
    def __init__(self) -> None:
        self._agent_policy: Any = SimpleNamespace(per_action_limit=lambda _action: None)
        self._registry: Any = SimpleNamespace(
            get=lambda name: SimpleNamespace(side_effecting=name.endswith("process_refund"))
        )
        self._evidence = GroundingEvidence(self._registry, self._agent_policy)


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

    def test_deterministic_checks_are_a_gate_not_evidence(self) -> None:
        """Pre-flight prose about the state *before* the action grounds no outcome."""
        checks = {
            "findings": [
                {
                    "check": "ownership",
                    "passed": True,
                    "reason": "Order exists and belongs to the authenticated customer.",
                }
            ]
        }
        evidence = _Support()._grounding_evidence(_state(deterministic_checks=checks))
        assert not any("authenticated customer" in chunk.content for chunk in evidence)


class TestALoopStepIsNamedForWhatItIs:
    def test_the_step_label_does_not_claim_react(self) -> None:
        support = _Support()
        support._steps = SimpleNamespace(step=3)  # type: ignore[assignment]
        support._max_rounds = 5
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
        """A DeterministicCheck line once "contradicted" the second sentence at 0.99."""
        checks = {
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
            deterministic_checks=checks,
        )
        reply = "Your refund for order 10829 has been completed. $32.94 has been refunded."
        result = await _guard().enforce(reply, support._grounding_evidence(state))
        assert not result.escalated, result.claim_checks
        assert result.response == reply


def _card(check: dict[str, Any]) -> str:
    from rich.text import Text

    from agent_platform.interface.tui.widgets import _append_claim

    body = Text()
    _append_claim(body, check)
    return body.plain.strip()


def _check(label: str, score: float, excerpt: str, *, model: str = "nli-test") -> EvidenceCheck:
    """One evidence line; a model-read one is the line the verdict was chosen from."""
    return EvidenceCheck(
        rank=1, similarity=0.9, tier="tool_result", excerpt=excerpt, nli_label=label,
        nli_score=score, decisive=bool(model) or label == "record", model=model,
        chosen=bool(model), source="get_order_status",
    )  # fmt: skip


def _verdict_card(
    supported: bool,
    tier: str,
    checks: list[EvidenceCheck],
    *,
    undetermined: bool = False,
    rationale: str = "",
) -> str:
    """The card a verdict draws, through `claim_view` — the path every agent's claims take."""
    verdict = DetectionVerdict(
        claim="the claim", supported=supported, undetermined=undetermined, confidence=0.9,
        tier=tier, detector_model="nli-test", evidence_checks=checks, rationale=rationale,
    )  # fmt: skip
    return _card(claim_check(verdict))


class TestTheCardShowsOnlyWhatDecidedIt:
    """Through the real projection: a verdict, as `claim_view` sends it, as the card draws it."""

    def test_a_model_verdict_keeps_its_probability(self) -> None:
        card = _verdict_card(True, "tool_result", [_check("entailment", 0.99, "hear back soon")])
        assert "entailment 0.990" in card and '"hear back soon"' in card

    def test_a_verdict_the_record_decided_names_the_line_not_the_rule(self) -> None:
        line = _check("record", 1.0, "Order 10665 refund status: pending_review.", model="")
        rationale = "refund status is pending_review, so a claim of pending_review holds"
        card = _verdict_card(True, DIRECT_TIER, [line], rationale=rationale)
        assert "Order 10665 refund status: pending_review." in card and "so a claim" not in card

    def test_a_contradiction_by_a_model_says_how_sure(self) -> None:
        card = _verdict_card(False, "tool_result", [_check("contradiction", 0.98, "refund: none")])
        assert "contradiction 0.980" in card and "[CONTRADICTED]" in card

    def test_a_rule_that_found_no_line_says_what_it_found(self) -> None:
        card = _card(
            {
                "claim": "$99.00 has been refunded",
                "verdict": "refuted",
                "rationale": "Amount(s) ['$99.00'] appear in the reply but in no tool result.",
                "evidence": [],
            }
        )
        assert "contradicted: Amount(s)" in card and "By [" not in card


class TestAnUnverifiedClaimIsOneLine:
    def test_evidence_too_far_to_be_scored_is_not_printed(self) -> None:
        far = [_check("not checked (below 0.3)", 0.0, "You do not decide which", model="")]
        card = _verdict_card(False, NO_TIER, far, undetermined=True)
        assert "no related evidence" in card
        assert "0.00" not in card and "not checked" not in card.lower()

    def test_a_line_the_model_did_score_is_named(self) -> None:
        line = _check("unknown", 0.98, "Refunded $21.35 for order 10001")
        card = _verdict_card(False, NO_TIER, [line], undetermined=True)
        assert "neutral 0.980" in card and "Refunded $21.35 for order 10001" in card


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

        from agent_platform.interface.tui.widgets import _append_claim

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
            '└ Tool result · process_refund: "Order 10665 refund status: pending_review."'
            in body.plain
        )
        # A record read directly needs no model named: the line says it is the record.
        assert "[SUPPORTED]" in body.plain and "Supported by" not in body.plain


@pytest.mark.usefixtures("abstaining_guard")
class TestAPromiseIsReadAsAPromise:
    """`cancel order 10193`: "will be processed once approved" was read as already processed."""

    def _turn(self) -> AgentState:
        return _state(
            (
                "support__cancel_order",
                "Order 10193 has been cancelled. Refund of $66.56 for order 10193 has been "
                "queued for human review. Order 10193 refund status: pending_review.",
                "referred",
            )
        )

    def _guard(self) -> ResponseGuardrail:
        return ResponseGuardrail(
            DetectionLadder(create_scoring_cascade(), entailment=_NeverReadsARecord())
        )

    async def test_the_reply_from_the_live_run_is_sent_whole(self) -> None:
        reply = (
            "Your order 10193 has been cancelled. The $66.56 refund is queued for review "
            "and will be processed once approved."
        )
        result = await self._guard().enforce(reply, _Support()._grounding_evidence(self._turn()))
        assert not result.escalated, result.claim_checks
        assert result.response == reply

    async def test_a_refund_stated_as_done_is_cut_and_the_rest_sent(self) -> None:
        reply = "Your order 10193 has been cancelled. The $66.56 refund has been processed."
        result = await self._guard().enforce(reply, _Support()._grounding_evidence(self._turn()))
        assert result.escalated and not result.withheld
        assert result.response == "Your order 10193 has been cancelled."
        assert result.removed_sentences == ["The $66.56 refund has been processed."]
        assert result.total_sentences == 2
        assert result.outcome.startswith("TRIMMED · removed 1 of 2 sentence(s)")

    async def test_nothing_left_is_withheld_and_blocked(self) -> None:
        reply = "The $66.56 refund has been processed."
        result = await self._guard().enforce(reply, _Support()._grounding_evidence(self._turn()))
        assert result.withheld
        assert result.response == get_settings().hallucination_check.withheld_reply
        assert result.outcome.startswith("BLOCKED · ") and result.outcome.endswith("reply withheld")


class TestTheCardIsDrawnLikeTheOthers:
    def _body(self, outcome: str) -> Any:
        from rich.text import Text

        from agent_platform.interface.tui.widgets import _append_grounding

        body = Text()
        _append_grounding(body, {"outcome": outcome, "claim_checks": []})
        return body

    def test_only_the_state_word_takes_the_outcome_colour(self) -> None:
        body = self._body("TRIMMED · removed 1 of 2 sentence(s) · a reason")
        assert body.plain.startswith("  result = TRIMMED · removed")
        styles = {body.plain[span.start : span.end]: str(span.style) for span in body.spans}
        tones = get_settings().tui.tone_styles
        assert styles["TRIMMED"] == tones["warn"]
        assert styles[" · removed 1 of 2 sentence(s) · a reason"] != tones["fail"]

    def test_a_claim_is_a_check_line_with_its_verdict(self) -> None:
        line = _check("record", 1.0, "Order 10193 refund status: pending_review.", model="")
        card = _verdict_card(False, DIRECT_TIER, [line])
        assert card.startswith("✗ the claim [CONTRADICTED]")


class TestTheCardNamesTheSource:
    """RA's evidence is a page; the card said `[Tool Result]` and never which one."""

    def test_a_web_page_is_named_by_its_url(self) -> None:
        line = _check("entailment", 0.97, "Hiring rose 4% in 2025")
        line = line.model_copy(update={"source": "https://www.wef.org/reports/future-of-jobs-2025"})
        card = _verdict_card(True, "tool_result", [line])
        assert "www.wef.org/reports/future-of-jobs-2025" in card

    async def test_the_resolver_carries_the_source_through(self) -> None:
        from agent_platform.context.types import Chunk

        chunk = Chunk(
            content="Order 10193 refund status: pending_review.",
            kind="tool_result",
            source="get_order_status",
            metadata={"observation_sequence": 0},
        )
        result = (
            await TestAPromiseIsReadAsAPromise()
            ._guard()
            .enforce("Your refund for order 10193 is pending review.", [chunk])
        )
        sources = [e["source"] for c in result.claim_checks for e in c["evidence"]]
        assert "get_order_status" in sources


class _LimitedSupport(_Support):
    """The agent with a $50.00 direct refund limit, which the guard (never the model) holds."""

    def __init__(self) -> None:
        super().__init__()
        from agent_platform.core.primitives.money import Money

        self._agent_policy = SimpleNamespace(per_action_limit=lambda _action: Money.parse("50.00"))
        self._evidence = GroundingEvidence(self._registry, self._agent_policy)


@pytest.mark.usefixtures("abstaining_guard")
class TestAReviewIsPromisedOnlyOnceOneIsQueued:
    """`refund order 10533` ($103.39, limit $50.00): a review was promised and none was queued.

    The model, told the limit, declined to call `process_refund`, so the
    Gatekeeper never escalated and nothing reached the queue. The guard cut
    the review sentence for the wrong reason ("$50.00 in no tool result",
    though the limit is policy evidence) and let "You'll hear back soon"
    through as unverified.
    """

    _READ = (
        "support__get_order_status",
        "Order 10533 placed 2026-08-20 has shipped and was delivered on 2026-08-28, $103.39. "
        "Order 10533 refund status: none",
    )
    _REFERRED = (
        "support__process_refund",
        "Refund of $103.39 for order 10533 has been queued for human review. "
        "Order 10533 refund status: pending_review",
        "referred",
    )
    _GUARD = staticmethod(
        lambda: ResponseGuardrail(
            DetectionLadder(create_scoring_cascade(), entailment=_NeverReadsARecord())
        )
    )

    async def test_with_nothing_queued_both_promises_are_cut_for_what_the_record_says(
        self,
    ) -> None:
        reply = (
            "Order 10533 was delivered on 2026-08-28 with a total of $103.39. Since the order "
            "total exceeds the $50.00 refund limit I'm authorized to handle directly, this refund "
            "request will be sent for human review. You'll hear back soon."
        )
        evidence = _LimitedSupport()._grounding_evidence(_state(self._READ))
        result = await self._GUARD().enforce(reply, evidence)
        assert result.response == "Order 10533 was delivered on 2026-08-28 with a total of $103.39."
        assert "You'll hear back soon." in result.removed_sentences
        refuted = [c for c in result.claim_checks if c["verdict"] == "refuted"]
        assert refuted and all("refund status is none" in c["rationale"] for c in refuted)
        assert result.reason_code == "TOOL_EVIDENCE_RECORD_STATE_MISMATCH"

    async def test_once_the_referral_ran_the_same_promises_are_supported(self) -> None:
        reply = (
            "Your refund request for order 10533 has been sent for human review. "
            "You'll hear back soon."
        )
        evidence = _LimitedSupport()._grounding_evidence(_state(self._READ, self._REFERRED))
        result = await self._GUARD().enforce(reply, evidence)
        assert not result.escalated, result.claim_checks
        assert result.response == reply


class TestAPolicyAmountIsNotAnUnsourcedOne:
    """The limit is in the guard's policy evidence; "$50.00 in no tool result" was wrong."""

    async def test_the_limit_is_found_in_policy(self) -> None:
        evidence = _LimitedSupport()._grounding_evidence(
            _state(TestAReviewIsPromisedOnlyOnceOneIsQueued._READ)
        )
        result = await TestAReviewIsPromisedOnlyOnceOneIsQueued._GUARD().enforce(
            "Refunds above $50.00 for order 10533 need additional review.", evidence
        )
        assert result.reason_code != "TOOL_EVIDENCE_AMOUNT_MISMATCH", result.claim_checks

    async def test_a_policy_amount_stated_as_money_moved_is_still_refuted(self) -> None:
        evidence = _LimitedSupport()._grounding_evidence(
            _state(TestAReviewIsPromisedOnlyOnceOneIsQueued._READ)
        )
        result = await TestAReviewIsPromisedOnlyOnceOneIsQueued._GUARD().enforce(
            "Your refund of $50.00 for order 10533 has been issued.", evidence
        )
        assert result.escalated and result.withheld

    async def test_an_amount_in_no_evidence_at_all_is_still_unsourced(self) -> None:
        evidence = _LimitedSupport()._grounding_evidence(
            _state(TestAReviewIsPromisedOnlyOnceOneIsQueued._READ)
        )
        result = await TestAReviewIsPromisedOnlyOnceOneIsQueued._GUARD().enforce(
            "Order 10533 cost $77.10.", evidence
        )
        assert result.reason_code == "TOOL_EVIDENCE_AMOUNT_MISMATCH"


class TestTheKnowledgeArticlesAreOneSentencePerPremise:
    """A whole KB section read as contradicting "I can assist with a refund for this delivered
    order": it also says an order in flight has nothing to refund. One rule per premise."""

    def test_no_premise_is_a_paragraph(self) -> None:
        from customer_support_agent.grounding.evidence import (
            knowledge_evidence as _knowledge_evidence,
        )

        chunks = _knowledge_evidence()
        assert chunks and all("\n" not in chunk.content for chunk in chunks)

    def test_what_may_be_refunded_and_what_may_not_are_separate_premises(self) -> None:
        from customer_support_agent.grounding.evidence import (
            knowledge_evidence as _knowledge_evidence,
        )

        contents = [chunk.content for chunk in _knowledge_evidence()]
        delivered = next(c for c in contents if "delivered order can be refunded" in c)
        assert "nothing to refund" not in delivered
