"""Scoring an intent classifier against labelled messages: hand-computed expectations."""

from __future__ import annotations

import pytest

from agent_platform.evals.intent_bakeoff import (
    LabelledMessage,
    Prediction,
    predict,
    render_report,
    score,
)


def _p(expected: str, predicted: str, confidence: float, latency_ms: float = 10.0) -> Prediction:
    return Prediction(
        text="",
        expected=expected,
        predicted=predicted,
        confidence=confidence,
        latency_ms=latency_ms,
    )


PREDICTIONS = [
    _p("A", "A", 0.9, 10.0),
    _p("A", "B", 0.6, 20.0),
    _p("B", "B", 0.8, 30.0),
    _p("C", "C", 0.4, 40.0),
]


def test_accuracy_and_per_label_recall() -> None:
    result = score(PREDICTIONS, threshold=0.75, bins=2)

    assert result.n == 4
    assert result.accuracy == pytest.approx(0.75)
    assert result.recall == {"A": 0.5, "B": 1.0, "C": 1.0}


def test_macro_f1_averages_every_label_equally() -> None:
    # A: p=1, r=.5 → .667 · B: p=.5, r=1 → .667 · C: 1
    result = score(PREDICTIONS, threshold=0.75, bins=2)

    assert result.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_confusion_counts_expected_against_predicted() -> None:
    result = score(PREDICTIONS, threshold=0.75, bins=2)

    assert result.confusion["A"] == {"A": 1, "B": 1}
    assert result.confusion["C"] == {"C": 1}


def test_coverage_is_the_share_the_threshold_lets_through() -> None:
    result = score(PREDICTIONS, threshold=0.75, bins=2)

    assert result.coverage == pytest.approx(0.5)
    assert result.accuracy_when_covered == pytest.approx(1.0)


def test_brier_and_expected_calibration_error_of_the_top_score() -> None:
    # Brier: (.1² + .6² + .2² + .6²) / 4
    # ECE, two bins: [0,.5) holds .4 (right) → |1-.4|·1/4; [.5,1] holds .9 .6 .8 (2/3 right)
    #   → |2/3 - .7667|·3/4
    result = score(PREDICTIONS, threshold=0.75, bins=2)

    assert result.brier == pytest.approx((0.01 + 0.36 + 0.04 + 0.36) / 4)
    assert result.ece == pytest.approx(0.6 * 0.25 + abs(2 / 3 - 2.3 / 3) * 0.75)


def test_latency_percentiles() -> None:
    result = score(PREDICTIONS, threshold=0.75, bins=2)

    assert result.p50_ms == pytest.approx(25.0)
    assert result.p95_ms == pytest.approx(38.5)


def test_an_empty_run_is_refused() -> None:
    with pytest.raises(ValueError):
        score([], threshold=0.75, bins=2)


@pytest.mark.asyncio
async def test_predict_runs_every_message_and_times_it() -> None:
    async def classify(text: str) -> tuple[str, float]:
        return ("B" if "b" in text else "A", 0.5)

    cases = [LabelledMessage(text="a", intent="A"), LabelledMessage(text="b", intent="A")]
    predictions = await predict(cases, classify)

    assert [(p.expected, p.predicted) for p in predictions] == [("A", "A"), ("A", "B")]
    assert all(p.latency_ms >= 0.0 for p in predictions)


def test_report_shows_each_contender_and_its_misses() -> None:
    scores = {"bert": score(PREDICTIONS, threshold=0.75, bins=2)}
    misses = {"bert": [PREDICTIONS[1]]}

    report = render_report("Intent bake-off", scores, misses, z=1.96)

    assert "bert" in report
    assert "3/4" in report
    assert "| A | B |" in report.replace("  ", " ")


def test_sweep_reads_coverage_and_accuracy_at_each_threshold() -> None:
    result = score(PREDICTIONS, threshold=0.75, bins=2, sweep=[0.5, 0.85])

    assert [(s.threshold, s.coverage, s.accuracy_when_covered) for s in result.sweep] == [
        (0.5, 0.75, pytest.approx(2 / 3)),
        (0.85, 0.25, 1.0),
    ]


def test_report_lists_the_sweep() -> None:
    scores = {"bert": score(PREDICTIONS, threshold=0.75, bins=2, sweep=[0.5])}

    report = render_report("Intent bake-off", scores, {}, z=1.96)

    assert "## Coverage by threshold" in report
    assert "0.75 / 0.667" in report


def test_each_contender_reports_its_own_threshold() -> None:
    scores = {
        "strict": score(PREDICTIONS, threshold=0.75, bins=2),
        "loose": score(PREDICTIONS, threshold=0.5, bins=2),
    }

    report = render_report("Intent bake-off", scores, {}, z=1.96)
    rows = {
        line.split("|")[1].strip(): line
        for line in report.splitlines()
        if line.startswith("| s") or line.startswith("| l")
    }

    assert scores["loose"].threshold == 0.5
    assert "0.750" in rows["strict"] and "0.500" in rows["strict"]
    assert "0.500" in rows["loose"] and "0.750" in rows["loose"]
