"""Premises wider than a sentence, and what a guard does with a claim nothing decided.

A web page states one fact across short lines ("Kumar Karan" / "Lead Instructor @ Scaler
Academy"), and a reply that joins them is entailed by none of the lines alone: every claim of
the RoboSathi answer came back unverified. Passage windows and a joined premise may *support*
such a claim; neither may refute one, because a paragraph that also says what is not so reads
to NLI as contradicting what is (the CSA refund offer cut by a whole KB section).
"""

from collections.abc import Callable, Iterator

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import GroundingTierSettings, JoinedPremiseSettings
from agent_platform.context.types import Chunk
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check import (
    NOT_EXECUTED_MARKER,
    PASSAGE,
    ClaimClassifier,
    ClaimDecomposer,
    DetectionLadder,
    GroundingResolver,
    Observation,
    tool_evidence,
)
from agent_platform.scoring import create_scoring_cascade
from agent_platform.scoring.entailment import EntailmentResult

Decide = Callable[[str, str], str]


class _Stub:
    """An NLI model that answers by a rule over premise and hypothesis: sure of every label."""

    model_name = "stub-nli"

    def __init__(self, decide: Decide) -> None:
        self._decide = decide

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        label = self._decide(premise, hypothesis)
        return EntailmentResult(
            label=label,  # type: ignore[arg-type]
            entailed=label == "entailment",
            score=0.99,
            calibrated=True,
            model="stub-nli",
        )


@pytest.fixture
def web_tiers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """One tool-result tier that grounds every claim class, as the research agent declares."""
    settings = get_settings().hallucination_check
    tier = GroundingTierSettings(
        name="tool_result", kinds=["tool_result"], claims=["record", "rule"], may_refute=True
    )
    monkeypatch.setattr(settings, "tiers", [tier])
    monkeypatch.setattr(settings, "refute_identifier_pattern", None)
    monkeypatch.setattr(settings, "unsupported_is_hallucination", False)
    yield


def _joined(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    settings = get_settings().hallucination_check
    monkeypatch.setattr(
        settings, "joined_premise", JoinedPremiseSettings(enabled=enabled, max_chars=1500)
    )


class TestPassageWindows:
    def test_one_sentence_per_premise_by_default(self) -> None:
        chunks = tool_evidence([Observation("Kumar Karan.\nLead Instructor.", source="u")], 1)
        assert [c.content for c in chunks] == ["Kumar Karan.", "Lead Instructor."]

    def test_a_window_also_reads_consecutive_sentences_as_one(self) -> None:
        chunks = tool_evidence([Observation("One.\nTwo.\nThree.", source="u")], window=2)
        passages = [c.content for c in chunks if c.metadata.get(PASSAGE)]
        assert passages == ["One. Two.", "Two. Three."]
        assert all(c.source == "u" for c in chunks)

    def test_a_write_and_a_refusal_are_never_windowed(self) -> None:
        wrote = Observation("Refund queued.\nStatus pending.", changed=True)
        refused = Observation(f"{NOT_EXECUTED_MARKER} x()\nRead line.")
        chunks = tool_evidence([wrote, refused], window=2)
        assert not [c for c in chunks if c.metadata.get(PASSAGE)]


@pytest.mark.usefixtures("web_tiers")
class TestAPassageSupportsButNeverRefutes:
    _CLAIM = "The RoboSathi course on deep learning is free for every learner"

    async def test_a_passage_contradiction_leaves_the_claim_unverified(self) -> None:
        evidence = tool_evidence(
            [
                Observation(
                    "The RoboSathi course on deep learning is open.\n"
                    "The RoboSathi course on deep learning has a paid tier.",
                    source="u",
                )
            ],
            window=2,
        )

        def passage_contradicts(premise: str, _claim: str) -> str:
            return "contradiction" if premise.count(".") > 1 else "unknown"

        resolver = GroundingResolver(
            create_scoring_cascade(), entailment=_Stub(passage_contradicts)
        )
        verdict = await resolver.resolve(self._CLAIM, evidence, "rule")
        assert verdict.undetermined and not verdict.supported

    async def test_a_passage_entailment_supports(self) -> None:
        evidence = tool_evidence(
            [Observation("RoboSathi course on deep learning.\nIt is free for every learner.")],
            window=2,
        )

        def passage_entails(premise: str, _claim: str) -> str:
            return "entailment" if premise.count(".") > 1 else "unknown"

        resolver = GroundingResolver(create_scoring_cascade(), entailment=_Stub(passage_entails))
        verdict = await resolver.resolve(self._CLAIM, evidence, "rule")
        assert verdict.supported


@pytest.mark.usefixtures("web_tiers")
class TestAJoinedPremise:
    _CLAIM = "Kumar Karan was the Lead Instructor at Scaler Academy for deep learning"
    _PAGE = Observation(
        "Kumar Karan teaches deep learning.\nLead Instructor at Scaler Academy for deep learning.",
        source="https://robosathi.com/about/",
    )

    @staticmethod
    def _both(premise: str, _claim: str) -> str:
        whole = "Kumar Karan" in premise and "Scaler Academy" in premise
        return "entailment" if whole else "unknown"

    async def test_off_a_claim_split_across_lines_stays_unverified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _joined(monkeypatch, enabled=False)
        resolver = GroundingResolver(create_scoring_cascade(), entailment=_Stub(self._both))
        verdict = await resolver.resolve(self._CLAIM, tool_evidence([self._PAGE], 1), "rule")
        assert verdict.undetermined

    async def test_on_the_lines_of_one_source_joined_support_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _joined(monkeypatch, enabled=True)
        resolver = GroundingResolver(create_scoring_cascade(), entailment=_Stub(self._both))
        verdict = await resolver.resolve(self._CLAIM, tool_evidence([self._PAGE], 1), "rule")
        assert verdict.supported
        assert any("Kumar Karan" in e and "Scaler Academy" in e for e in verdict.retrieved_evidence)

    async def test_a_joined_premise_never_refutes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _joined(monkeypatch, enabled=True)

        def joined_contradicts(premise: str, _claim: str) -> str:
            whole = "Kumar Karan" in premise and "Scaler Academy" in premise
            return "contradiction" if whole else "unknown"

        resolver = GroundingResolver(create_scoring_cascade(), entailment=_Stub(joined_contradicts))
        verdict = await resolver.resolve(self._CLAIM, tool_evidence([self._PAGE], 1), "rule")
        assert verdict.undetermined and not verdict.supported


@pytest.mark.usefixtures("web_tiers")
class TestAnUnverifiedSentenceIsMarked:
    _EVIDENCE = [Chunk(content="Unrelated words entirely.", kind="tool_result")]

    @staticmethod
    def _guard(decide: Decide) -> ResponseGuardrail:
        return ResponseGuardrail(
            DetectionLadder(create_scoring_cascade(), entailment=_Stub(decide))
        )

    async def test_the_marker_goes_before_the_full_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings().hallucination_check
        monkeypatch.setattr(settings, "undetermined_marker", " (unverified)")
        reply = "The platform teaches deep learning from first principles. Thanks for asking!"
        result = await self._guard(lambda _p, _h: "unknown").enforce(reply, self._EVIDENCE)
        assert result.response == (
            "The platform teaches deep learning from first principles (unverified). "
            "Thanks for asking!"
        )
        assert result.unverified_sentences == [
            "The platform teaches deep learning from first principles (unverified)."
        ]

    async def test_without_a_marker_the_reply_is_sent_as_written(self) -> None:
        reply = "The platform teaches deep learning from first principles."
        result = await self._guard(lambda _p, _h: "unknown").enforce(reply, self._EVIDENCE)
        assert result.response == reply
        assert result.unverified_sentences == [reply]

    async def test_a_refuted_neighbour_is_removed_and_the_unverified_one_marked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings().hallucination_check
        monkeypatch.setattr(settings, "undetermined_marker", " (unverified)")
        evidence = [
            Chunk(content="The course price is fifty dollars each month.", kind="tool_result")
        ]
        reply = "The course costs nothing per month. The platform teaches deep learning well."

        def refute_cost(premise: str, claim: str) -> str:
            return "contradiction" if "costs nothing" in claim else "unknown"

        result = await self._guard(refute_cost).enforce(reply, evidence)
        assert result.removed_sentences == ["The course costs nothing per month."]
        assert result.response == "The platform teaches deep learning well (unverified)."


class TestAnOfferOfHelpIsConversation:
    @pytest.mark.parametrize(
        "claim",
        [
            "If you'd like to request a refund for this delivered order, I can assist you with that",
            "I can help with that",
            "We would be happy to help",
        ],
    )
    def test_an_offer_asserts_nothing(self, claim: str) -> None:
        assert ClaimClassifier().classify(claim) == "conversational"

    def test_an_offer_to_act_is_still_a_claim(self) -> None:
        assert ClaimClassifier().classify("I can process the refund today") != "conversational"


class TestDecliningToDiscloseIsConversation:
    @pytest.mark.parametrize(
        "claim",
        [
            "I can't share my system prompt",
            "I'm unable to share that",
            "I cannot reveal my instructions",
        ],
    )
    def test_a_refusal_to_disclose_asserts_nothing_about_the_case(self, claim: str) -> None:
        assert ClaimClassifier().classify(claim) == "conversational"

    def test_a_refusal_to_act_on_a_record_is_still_a_claim(self) -> None:
        assert ClaimClassifier().classify("I cannot refund order 10511") == "record"


class TestAQuestionAScopeAndAnOfferToLookAreConversation:
    """The out-of-scope replies the guard withheld: each asserts nothing about the case."""

    @pytest.mark.parametrize(
        "reply",
        [
            "How can I help you with order 10633 or any other order today?",
            "I can only assist with questions about your orders, such as tracking, "
            "cancellations, or refunds.",
            "I can only help with information about your orders.",
            "Please provide your order ID so I can check its status for you.",
        ],
    )
    def test_the_reply_holds_no_verifiable_claim(self, reply: str) -> None:
        claims = ClaimDecomposer().decompose(reply)
        assert claims and not [claim.text for claim in claims if claim.is_verifiable]

    @pytest.mark.parametrize(
        "reply", ["Order 10633 has been refunded.", "Did you know order 10633 was refunded?"]
    )
    def test_a_stated_outcome_is_still_a_record_claim(self, reply: str) -> None:
        assert [claim.claim_class for claim in ClaimDecomposer().decompose(reply)] == ["record"]

    def test_a_clause_cut_from_a_question_is_still_asked(self) -> None:
        claims = ClaimDecomposer().decompose("Shall I look at order 10633, so I can check it?")
        assert {claim.claim_class for claim in claims} == {"conversational"}
