"""The second opinion on a claim NLI left undecided: it may confirm, never refute.

The RoboSathi answer: every fact of "Kumar Karan is a former Engineering Manager at Turing,
former Lead Instructor at Scaler Academy" was on the page, one per line, and NLI — reading one
premise at a time — said neutral to each. A decision model reads the claim beside the page.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import ClaimConfirmerSettings, GroundingTierSettings
from agent_platform.evals import create_scoring_cascade
from agent_platform.evals.entailment import EntailmentResult
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check import DetectionLadder, Observation, tool_evidence
from agent_platform.hallucination_check.confirmer import (
    Confirmation,
    LayaClaimConfirmer,
    create_claim_confirmer,
)
from agent_platform.hallucination_check.resolver import CONFIRM_TIER, GroundingResolver

ABOUT = Observation(
    "Kumar Karan\nEngineering Manager — AI @ Turing\nLead Instructor @ Scaler Academy",
    source="https://robosathi.com/about/",
)
CLAIM = "Kumar Karan is a former Engineering Manager at Turing and Lead Instructor at Scaler"


class _FakeLaya:
    """A loaded checkpoint answering every question with fixed probabilities."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities
        self.asked: list[dict[str, Any]] = []

    def system_one(self, state: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
        self.asked.append(state)
        qid = next(iter(question))
        return {"answers": {qid: {"probabilities": dict(self._probabilities)}}}


def _settings(tmp_path: Path, **overrides: Any) -> ClaimConfirmerSettings:
    folder = tmp_path / "laya"
    folder.mkdir(exist_ok=True)
    for name in ("model.safetensors", "rl_agent_config.json"):
        (folder / name).write_text("x")
    base = get_settings().hallucination_check.confirmer
    return base.model_copy(
        update={"enabled": True, "model": "org/laya", "subfolder": "decisions",
                "local_model_dir": str(folder), **overrides}
    )  # fmt: skip


def _confirmer(tmp_path: Path, probabilities: dict[str, float]) -> tuple[Any, _FakeLaya]:
    agent = _FakeLaya(probabilities)
    loads: list[Path] = []

    def loader(folder: Path, device: str) -> _FakeLaya:
        loads.append(folder)
        return agent

    return LayaClaimConfirmer(_settings(tmp_path), loader=loader), agent


class TestTheConfirmer:
    async def test_its_choice_of_supported_above_the_bar_confirms(self, tmp_path: Path) -> None:
        confirmer, agent = _confirmer(
            tmp_path, {"supported": 0.6, "contradicted": 0.1, "not_stated": 0.3}
        )
        answer = await confirmer.confirm(CLAIM, "evidence text")
        assert answer == Confirmation(confirmed=True, score=0.6, label="supported")
        words = get_settings().hallucination_check.confirmer
        assert agent.asked == [{words.claim_key: CLAIM, words.evidence_key: "evidence text"}]

    async def test_supported_below_the_bar_does_not_confirm(self, tmp_path: Path) -> None:
        confirmer, _ = _confirmer(
            tmp_path, {"supported": 0.4, "contradicted": 0.25, "not_stated": 0.35}
        )
        assert not (await confirmer.confirm(CLAIM, "e")).confirmed

    async def test_any_other_choice_does_not_confirm(self, tmp_path: Path) -> None:
        confirmer, _ = _confirmer(
            tmp_path, {"supported": 0.3, "contradicted": 0.2, "not_stated": 0.5}
        )
        answer = await confirmer.confirm(CLAIM, "e")
        assert not answer.confirmed and answer.label == "not_stated"

    def test_off_unless_an_agent_turns_it_on(self, tmp_path: Path) -> None:
        assert create_claim_confirmer(_settings(tmp_path, enabled=False)) is None
        assert isinstance(create_claim_confirmer(_settings(tmp_path)), LayaClaimConfirmer)


class _Scripted:
    """A confirmer that answers as told and records the evidence it was shown."""

    model_name = "stub-confirmer"

    def __init__(self, confirmed: bool) -> None:
        self._confirmed = confirmed
        self.read: list[str] = []

    async def confirm(self, claim: str, evidence: str) -> Confirmation:
        self.read.append(evidence)
        label = "supported" if self._confirmed else "not_stated"
        return Confirmation(confirmed=self._confirmed, score=0.6, label=label)


class _Nli:
    """NLI as on the RoboSathi turn: neutral to every single line, or contradicting all."""

    model_name = "stub-nli"

    def __init__(self, label: str = "unknown") -> None:
        self._label = label

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        return EntailmentResult(
            label=self._label,  # type: ignore[arg-type]
            entailed=False,
            score=0.99,
            calibrated=True,
            model="stub-nli",
        )


@pytest.fixture
def web_tiers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    settings = get_settings().hallucination_check
    tier = GroundingTierSettings(
        name="tool_result", kinds=["tool_result"], claims=["record", "rule"], may_refute=True
    )
    monkeypatch.setattr(settings, "tiers", [tier])
    monkeypatch.setattr(settings, "refute_identifier_pattern", None)
    monkeypatch.setattr(settings, "unsupported_is_hallucination", False)
    yield


def _resolver(confirmer: _Scripted, nli: _Nli | None = None) -> GroundingResolver:
    return GroundingResolver(create_scoring_cascade(), nli or _Nli(), confirmer=confirmer)


@pytest.mark.usefixtures("web_tiers")
class TestTheSecondOpinionInTheCascade:
    async def test_a_claim_nli_left_undecided_is_confirmed_from_its_source(self) -> None:
        confirmer = _Scripted(confirmed=True)
        verdict = await _resolver(confirmer).resolve(CLAIM, tool_evidence([ABOUT], 1), "rule")
        assert verdict.supported and verdict.tier == CONFIRM_TIER
        # The lines it read are the page's own, in the page's order.
        assert confirmer.read[0].splitlines()[0] == "Kumar Karan"

    async def test_a_claim_it_does_not_confirm_stays_unverified_and_says_why(self) -> None:
        confirmer = _Scripted(confirmed=False)
        verdict = await _resolver(confirmer).resolve(CLAIM, tool_evidence([ABOUT], 1), "rule")
        assert verdict.undetermined and "stub-confirmer: not_stated" in verdict.rationale

    async def test_it_is_never_asked_about_a_claim_nli_refuted(self) -> None:
        confirmer = _Scripted(confirmed=True)
        resolver = _resolver(confirmer, _Nli("contradiction"))
        verdict = await resolver.resolve(CLAIM, tool_evidence([ABOUT], 1), "rule")
        assert not verdict.supported and not verdict.undetermined
        assert confirmer.read == []

    async def test_the_source_text_stays_within_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings().hallucination_check
        small = settings.confirmer.model_copy(update={"max_evidence_chars": 40})
        monkeypatch.setattr(settings, "confirmer", small)
        confirmer = _Scripted(confirmed=False)
        await _resolver(confirmer).resolve(CLAIM, tool_evidence([ABOUT], 1), "rule")
        assert all(len(text) <= 40 for text in confirmer.read)

    async def test_the_card_names_the_second_opinion(self) -> None:
        guard = ResponseGuardrail(
            DetectionLadder(create_scoring_cascade(), entailment=_Nli(), confirmer=_Scripted(True))
        )
        result = await guard.enforce(f"{CLAIM}.", tool_evidence([ABOUT], 1))
        words = get_settings().hallucination_check.methods
        assert result.claim_checks[0]["method"] == words.confirmed.format(
            model="stub-confirmer", prob=0.6
        )
        assert result.claim_checks[0]["basis"] == ""


class TestPageTextIsReadAsSentences:
    @pytest.fixture(autouse=True)
    def pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings().hallucination_check
        monkeypatch.setattr(settings, "tool_results_are_pages", True)

    def test_a_wrapped_sentence_is_one_premise(self) -> None:
        page = Observation(
            "I built RoboSathi because I could not find a single free resource\n"
            "that taught AI from genuine first principles.\nKumar Karan",
            source="u",
        )
        assert [c.content for c in tool_evidence([page], 1)] == [
            "I built RoboSathi because I could not find a single free resource that taught AI "
            "from genuine first principles.",
            "Kumar Karan",
        ]

    def test_a_heading_and_its_repeat_are_read_once(self) -> None:
        page = Observation("### Deep Learning Fundamentals\nDeep Learning Fundamentals", source="u")
        assert [c.content for c in tool_evidence([page], 1)] == ["Deep Learning Fundamentals"]

    def test_records_keep_their_lines_as_written_when_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = get_settings().hallucination_check
        monkeypatch.setattr(settings, "tool_results_are_pages", False)
        page = Observation("Order 10042 was shipped\nand delivered", source="u")
        assert [c.content for c in tool_evidence([page], 1)] == [
            "Order 10042 was shipped",
            "and delivered",
        ]
