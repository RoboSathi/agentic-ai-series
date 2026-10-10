"""Scoring intent classifiers against labelled messages, side by side.

Agent-agnostic: an agent supplies its labelled messages and, per contender, a
`classify(text) -> (intent, confidence)` — a bare model, or its whole router.
Everything here is arithmetic over what came back: accuracy and macro-F1 for
how often it is right, coverage at the agent's threshold for how often it would
be trusted, Brier and ECE for whether its confidence means anything, and latency.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence

from pydantic import BaseModel

from agent_platform.evals.reporting.markdown import rate, table

Classify = Callable[[str], Awaitable[tuple[str, float]]]

_MS = 1000.0
_P50, _P95 = 0.50, 0.95


class LabelledMessage(BaseModel):
    """One message and the intent a person says it is."""

    text: str
    intent: str


class Prediction(BaseModel):
    """What one contender said about one message."""

    text: str
    expected: str
    predicted: str
    confidence: float
    latency_ms: float

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted


class ThresholdPoint(BaseModel):
    """How much a threshold lets through, and how often what it lets through is right."""

    threshold: float
    coverage: float
    accuracy_when_covered: float


class IntentScore(BaseModel):
    """One contender's results over the whole set."""

    # The routing threshold `coverage` was read at: each contender's own.
    threshold: float
    n: int
    correct: int
    accuracy: float
    macro_f1: float
    recall: dict[str, float]
    confusion: dict[str, dict[str, int]]
    coverage: float
    accuracy_when_covered: float
    brier: float
    ece: float
    p50_ms: float
    p95_ms: float
    sweep: list[ThresholdPoint] = []


async def predict(cases: Sequence[LabelledMessage], classify: Classify) -> list[Prediction]:
    """Classify every message in order, timing each call."""
    predictions = []
    for case in cases:
        started = time.perf_counter()
        intent, confidence = await classify(case.text)
        elapsed = (time.perf_counter() - started) * _MS
        predictions.append(
            Prediction(
                text=case.text,
                expected=case.intent,
                predicted=intent,
                confidence=confidence,
                latency_ms=elapsed,
            )
        )
    return predictions


def score(
    predictions: Sequence[Prediction],
    threshold: float,
    bins: int,
    sweep: Sequence[float] = (),
) -> IntentScore:
    """Every metric for one contender's predictions; `sweep` reads coverage at more thresholds."""
    if not predictions:
        raise ValueError("no predictions to score")
    correct = sum(p.correct for p in predictions)
    covered = _covered(predictions, threshold)
    latencies = sorted(p.latency_ms for p in predictions)
    return IntentScore(
        threshold=threshold,
        n=len(predictions),
        correct=correct,
        accuracy=correct / len(predictions),
        macro_f1=_macro_f1(predictions),
        recall=_recall(predictions),
        confusion=_confusion(predictions),
        coverage=len(covered) / len(predictions),
        accuracy_when_covered=_share_correct(covered),
        brier=sum((p.confidence - p.correct) ** 2 for p in predictions) / len(predictions),
        ece=_ece(predictions, bins),
        p50_ms=_quantile(latencies, _P50),
        p95_ms=_quantile(latencies, _P95),
        sweep=[_point(predictions, level) for level in sweep],
    )


def _covered(predictions: Sequence[Prediction], threshold: float) -> list[Prediction]:
    return [p for p in predictions if p.confidence >= threshold]


def _point(predictions: Sequence[Prediction], threshold: float) -> ThresholdPoint:
    covered = _covered(predictions, threshold)
    return ThresholdPoint(
        threshold=threshold,
        coverage=len(covered) / len(predictions),
        accuracy_when_covered=_share_correct(covered),
    )


def _share_correct(predictions: Sequence[Prediction]) -> float:
    return sum(p.correct for p in predictions) / len(predictions) if predictions else 0.0


def _recall(predictions: Sequence[Prediction]) -> dict[str, float]:
    by_label: dict[str, list[Prediction]] = defaultdict(list)
    for p in predictions:
        by_label[p.expected].append(p)
    return {label: _share_correct(items) for label, items in sorted(by_label.items())}


def _macro_f1(predictions: Sequence[Prediction]) -> float:
    """The mean of each label's F1, so a rare intent counts as much as a common one."""
    labels = {p.expected for p in predictions} | {p.predicted for p in predictions}
    scores = [_f1(predictions, label) for label in labels]
    return sum(scores) / len(scores)


def _f1(predictions: Sequence[Prediction], label: str) -> float:
    hits = sum(p.correct and p.expected == label for p in predictions)
    said = sum(p.predicted == label for p in predictions)
    actual = sum(p.expected == label for p in predictions)
    if not hits:
        return 0.0
    precision, recall = hits / said, hits / actual
    return 2 * precision * recall / (precision + recall)


def _confusion(predictions: Sequence[Prediction]) -> dict[str, dict[str, int]]:
    """Expected label → how often each label was predicted for it."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for p in predictions:
        counts[p.expected][p.predicted] += 1
    return {label: dict(counter) for label, counter in sorted(counts.items())}


def _ece(predictions: Sequence[Prediction], bins: int) -> float:
    """Expected calibration error: how far stated confidence sits from accuracy, by bin."""
    buckets: dict[int, list[Prediction]] = defaultdict(list)
    for p in predictions:
        buckets[min(int(p.confidence * bins), bins - 1)].append(p)
    return sum(
        abs(_share_correct(items) - sum(p.confidence for p in items) / len(items))
        * len(items)
        / len(predictions)
        for items in buckets.values()
    )


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between the closest ranks."""
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def render_report(
    title: str,
    scores: Mapping[str, IntentScore],
    misses: Mapping[str, Sequence[Prediction]],
    z: float,
) -> str:
    """The comparison as Markdown: summary, recall by intent, then every miss."""
    sections = [f"# {title}", _summary(scores, z), _recall_table(scores)]
    if any(s.sweep for s in scores.values()):
        sections.append(_sweep_table(scores))
    sections += [_misses(name, items) for name, items in misses.items()]
    return "\n\n".join(sections) + "\n"


def _summary(scores: Mapping[str, IntentScore], z: float) -> str:
    headers = [
        "contender",
        "accuracy",
        "macro-F1",
        "threshold",
        "trusted ≥ threshold",
        "accuracy when trusted",
        "Brier",
        "ECE",
        "p50 ms",
        "p95 ms",
    ]
    rows = [
        [
            name,
            rate(s.correct, s.n, z),
            s.macro_f1,
            s.threshold,
            s.coverage,
            s.accuracy_when_covered,
            s.brier,
            s.ece,
            s.p50_ms,
            s.p95_ms,
        ]
        for name, s in scores.items()
    ]
    return "## Summary\n\n" + table(headers, rows)


def _recall_table(scores: Mapping[str, IntentScore]) -> str:
    labels = sorted({label for s in scores.values() for label in s.recall})
    rows = [[label, *(s.recall.get(label) for s in scores.values())] for label in labels]
    return "## Recall by intent\n\n" + table(["intent", *scores], rows)


def _sweep_table(scores: Mapping[str, IntentScore]) -> str:
    """Per threshold, each contender's coverage / accuracy of what it covers."""
    levels = sorted({point.threshold for s in scores.values() for point in s.sweep})
    rows = [[level, *(_at(s, level) for s in scores.values())] for level in levels]
    heading = "## Coverage by threshold\n\nEach cell: share trusted / accuracy of that share.\n\n"
    return heading + table(["threshold", *scores], rows)


def _at(result: IntentScore, level: float) -> str:
    point = next((p for p in result.sweep if p.threshold == level), None)
    return f"{point.coverage:.2f} / {point.accuracy_when_covered:.3f}" if point else ""


def _misses(name: str, items: Sequence[Prediction]) -> str:
    rows = [[p.text, p.expected, p.predicted, p.confidence] for p in items]
    return f"## Misses · {name}\n\n" + table(["message", "expected", "predicted", "score"], rows)
