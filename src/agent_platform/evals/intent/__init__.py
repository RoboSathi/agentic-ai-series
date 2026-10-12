"""The intent bake-off: an agent's classifiers scored side by side (`STAGE=intent`)."""

from agent_platform.evals.intent.scoring import (
    BakeoffResult,
    IntentScore,
    LabelledMessage,
    Prediction,
    predict,
    render_report,
    score,
)

__all__ = [
    "BakeoffResult",
    "IntentScore",
    "LabelledMessage",
    "Prediction",
    "predict",
    "render_report",
    "score",
]
