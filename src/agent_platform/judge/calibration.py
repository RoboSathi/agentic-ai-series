"""How far a judge can be trusted: its agreement with a second rater (E11).

A judge that has never been measured against another rater is an opinion
with a temperature setting. `JudgeCalibrator` is what turns it into an
instrument: Cohen's kappa against a second rater's labels (a person, or
an independent check such as the runtime hallucination guard), and
pairwise comparison where absolute scores are known to be unreliable.
"""

import math

from pydantic import BaseModel, Field

from agent_platform.config import get_settings


class AgreementReport(BaseModel):
    """Agreement between a judge and a second rater, chance removed."""

    sample_size: int
    observed_agreement: float
    expected_agreement: float
    cohens_kappa: float
    acceptable: bool
    labels: dict[str, int] = Field(default_factory=dict)


class JudgeCalibrator:
    """Measures a judge against a second rater using Cohen's kappa.

    Raw agreement flatters any judge on a skewed label distribution —
    always answering "supported" scores 90% when 90% of claims are
    supported. Kappa removes the agreement that chance alone explains,
    which is why it, not accuracy, is the gate.
    """

    def __init__(self, acceptable: float | None = None) -> None:
        self._acceptable = (
            acceptable if acceptable is not None else get_settings().judge.kappa_acceptable
        )

    def agreement(self, judge_labels: list[str], rater_labels: list[str]) -> AgreementReport:
        """Cohen's kappa between the judge's labels and the second rater's."""
        if len(judge_labels) != len(rater_labels):
            raise ValueError("Judge and rater label sets must be the same length")
        total = len(judge_labels)
        if total == 0:
            return AgreementReport(
                sample_size=0,
                observed_agreement=0.0,
                expected_agreement=0.0,
                cohens_kappa=0.0,
                acceptable=False,
            )
        observed = sum(a == b for a, b in zip(judge_labels, rater_labels, strict=True)) / total
        expected = self._chance_agreement(judge_labels, rater_labels)
        kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
        return AgreementReport(
            sample_size=total,
            observed_agreement=observed,
            expected_agreement=expected,
            cohens_kappa=kappa,
            acceptable=kappa >= self._acceptable,
            labels=_label_counts(judge_labels),
        )

    @staticmethod
    def _chance_agreement(judge_labels: list[str], rater_labels: list[str]) -> float:
        """Agreement two raters would reach at random, given their label mixes."""
        total = len(judge_labels)
        judge_counts, rater_counts = _label_counts(judge_labels), _label_counts(rater_labels)
        return sum(
            (judge_counts.get(label, 0) / total) * (rater_counts.get(label, 0) / total)
            for label in set(judge_counts) | set(rater_counts)
        )

    @staticmethod
    def pairwise_win_rate(preferences: list[str], candidate: str) -> float:
        """Share of pairwise comparisons a candidate won.

        Pairwise preference survives judges whose absolute scores drift
        between runs: "which of these two is better" stays stable when
        "rate this out of ten" does not.
        """
        if not preferences:
            return 0.0
        return preferences.count(candidate) / len(preferences)

    @staticmethod
    def standard_error(win_rate: float, comparisons: int) -> float:
        """Binomial standard error — how much of a win rate is just sample size."""
        if comparisons <= 0:
            return 0.0
        return math.sqrt(max(0.0, win_rate * (1.0 - win_rate)) / comparisons)


def _label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
