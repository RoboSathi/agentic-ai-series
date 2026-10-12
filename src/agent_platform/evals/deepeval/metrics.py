"""Metric specs (an agent's `metrics.yaml`) as DeepEval conversational metrics (Factory).

Three kinds, each the standard DeepEval metric for a multi-turn agent:

- `conversational_geval` — one quality dimension, anchored: criteria, the
  steps a judge follows, and rubric bands saying what each score means;
- `turn_faithfulness` — every reply's claims checked against the evidence
  the agent was shown for it (the judge-side twin of the runtime guard);
- `turn_relevancy` — every reply answers what the customer just asked.
"""

from collections.abc import Callable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import QualityMetricSpec

GEVAL, FAITHFULNESS, RELEVANCY = "conversational_geval", "turn_faithfulness", "turn_relevancy"


class UnknownMetricKindError(ValueError):
    """A metric spec names a kind this factory does not build."""


class UnshowableParamError(ValueError):
    """A G-Eval spec asks to show the judge a field `ConversationalGEval` cannot show."""


def build_metrics(deepeval: Any, specs: list[QualityMetricSpec], model: Any) -> list[Any]:
    """One DeepEval metric per spec, every one graded by `model`."""
    builders: dict[str, Callable[[Any, QualityMetricSpec, Any], Any]] = {
        GEVAL: _geval,
        FAITHFULNESS: _faithfulness,
        RELEVANCY: _relevancy,
    }
    metrics = []
    for spec in specs:
        builder = builders.get(spec.kind)
        if builder is None:
            raise UnknownMetricKindError(
                f"'{spec.kind}' (metric {spec.name}); kinds: {', '.join(builders)}"
            )
        metrics.append(builder(deepeval, spec, model))
    return metrics


def _geval(deepeval: Any, spec: QualityMetricSpec, model: Any) -> Any:
    shown = [
        _shown_param(deepeval, spec.name, value)
        for value in spec.params or get_settings().evals.deepeval.geval_params
    ]
    rubric = [
        deepeval.metrics.g_eval.utils.Rubric(
            score_range=band.score_range, expected_outcome=band.expected_outcome
        )
        for band in spec.rubric
    ]
    return deepeval.metrics.ConversationalGEval(
        name=spec.name,
        criteria=spec.criteria or None,
        evaluation_steps=spec.evaluation_steps or None,
        evaluation_params=shown,
        rubric=rubric or None,
        threshold=spec.threshold,
        model=model,
        async_mode=True,
        _include_g_eval_suffix=False,
    )


def _shown_param(deepeval: Any, metric: str, value: str) -> Any:
    """`value` as a `MultiTurnParams`, refused now rather than as an error on every case."""
    showable = deepeval.metrics.g_eval.utils.CONVERSATIONAL_G_EVAL_PARAMS
    param = deepeval.test_case.MultiTurnParams(value)
    if param not in showable:
        names = ", ".join(sorted(item.value for item in showable))
        raise UnshowableParamError(f"'{value}' (metric {metric}); showable: {names}")
    return param


def _faithfulness(deepeval: Any, spec: QualityMetricSpec, model: Any) -> Any:
    return deepeval.metrics.TurnFaithfulnessMetric(
        threshold=spec.threshold,
        model=model,
        include_reason=get_settings().evals.deepeval.include_reason,
        async_mode=True,
    )


def _relevancy(deepeval: Any, spec: QualityMetricSpec, model: Any) -> Any:
    return deepeval.metrics.TurnRelevancyMetric(
        threshold=spec.threshold,
        model=model,
        include_reason=get_settings().evals.deepeval.include_reason,
        async_mode=True,
    )
