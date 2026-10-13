"""One claim, projected once for every card and log line (`hallucination_check/claim_view.py`)."""

from agent_platform.context.types import Chunk
from agent_platform.hallucination_check.claim_view import (
    claim_check,
    model_scores,
    provenance_report,
    tier_name,
    tiers_searched,
)
from agent_platform.hallucination_check.types import DetectionVerdict, EvidenceCheck


def _line(model: str, label: str, score: float, *, chosen: bool = False) -> EvidenceCheck:
    return EvidenceCheck(
        rank=1, similarity=0.9, tier="tool_result", excerpt="x", nli_label=label,
        nli_score=score, decisive=chosen, model=model, chosen=chosen,
    )  # fmt: skip


def test_each_model_is_shown_by_the_line_the_verdict_rests_on() -> None:
    # The surer neutral line loses to the one the verdict was chosen from.
    verdict = DetectionVerdict(
        claim="c",
        supported=True,
        confidence=0.9,
        evidence_checks=[_line("m", "unknown", 0.99), _line("m", "entailment", 0.8, chosen=True)],
    )
    assert list(model_scores(verdict).values()) == ["entailment 0.80"]


def test_the_nli_third_label_reads_as_neutral() -> None:
    verdict = DetectionVerdict(
        claim="c", supported=False, confidence=0.5, evidence_checks=[_line("m", "unknown", 0.7)]
    )
    assert list(model_scores(verdict).values()) == ["neutral 0.70"]


def test_tiers_are_named_from_config() -> None:
    assert tier_name("tool_result") == "tool result"
    assert tier_name("unheard_of") == "unheard_of"


def test_evidence_searched_and_provenance_are_worded_from_config() -> None:
    evidence = [Chunk(content="Order 10001 total $32.00", kind="tool_result")]
    assert tiers_searched(evidence) == "1 tool result"
    assert tiers_searched([]) == "no evidence"
    lines = provenance_report("Your refund of $32.00 is on its way.", evidence)
    assert lines and lines[0].startswith("✓") and "tool result" in lines[0]


def test_a_projected_claim_carries_its_steps() -> None:
    verdict = DetectionVerdict(
        claim="c", supported=True, confidence=0.9, tier="nli", detector_model="m",
        evidence_checks=[_line("m", "entailment", 0.9, chosen=True)],
    )  # fmt: skip
    projected = claim_check(verdict)
    assert projected["verdict"] == "supported" and projected["readings"]
    assert projected["evidence"][0]["tier"] == "tool result"
