"""The intent bake-off: every configured contender over the same labelled messages.

Each contender is a config profile (or none: the agent as configured). For each,
the classifier is built exactly as the agent builds it and scored twice:

- **model**: the classifier alone on every message, in this agent's intents —
  what it decides when it is asked;
- **router**: the whole front door (rules, then the model, then the threshold) —
  what the agent would actually do. A turn the router leaves to the LLM counts
  as UNKNOWN, so this row also measures how much work reaches the LLM.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from agent_platform.config import get_settings, reload_settings
from agent_platform.config.yaml_config import clear_overlays, use_overlay
from agent_platform.evals.intent import (
    BakeoffResult,
    IntentScore,
    LabelledMessage,
    Prediction,
    predict,
    score,
)
from agent_platform.intent import create_intent_classifier
from customer_support_agent.config import get_agent_settings
from customer_support_agent.config.settings import reload_agent_settings
from customer_support_agent.intent.router import CustomerSupportIntentRouter, intent_choices

MODEL, ROUTER = "model", "router"


class _CaseFile(BaseModel):
    cases: list[LabelledMessage]


def load_cases(path: Path) -> list[LabelledMessage]:
    """The labelled messages, validated."""
    return _CaseFile.model_validate(yaml.safe_load(path.read_text())).cases


async def run() -> BakeoffResult:
    """Score every contender, model alone and through the router (the platform writes the report)."""
    bakeoff = get_agent_settings().evals.intent_bakeoff
    cases = load_cases(bakeoff.resolved_cases_path)
    scores: dict[str, IntentScore] = {}
    misses: dict[str, list[Prediction]] = {}
    try:
        for name, profiles in bakeoff.contenders.items():
            for mode, predictions in (await _contender(profiles, cases)).items():
                label = f"{name} · {mode}"
                scores[label] = _score(predictions, sweep=mode == MODEL)
                misses[label] = [p for p in predictions if not p.correct]
    finally:
        _apply([])
    return BakeoffResult(title=bakeoff.report_title, scores=scores, misses=misses)


async def _contender(
    profiles: list[str], cases: list[LabelledMessage]
) -> dict[str, list[Prediction]]:
    """One contender's predictions in both modes, under its own config."""
    _apply(profiles)
    backend = create_intent_classifier(**intent_choices())
    backend.ensure_loaded_sync()
    router = CustomerSupportIntentRouter(backend=backend)

    async def model_only(text: str) -> tuple[str, float]:
        top = (await router.rank_with_model(text))[0]
        return top.intent, top.score

    async def routed(text: str) -> tuple[str, float]:
        intent = await router.classify_async(text)
        return intent.name, intent.confidence

    return {MODEL: await predict(cases, model_only), ROUTER: await predict(cases, routed)}


def _apply(profiles: list[str]) -> None:
    """Make these profiles (in order) the run's config layers, and rebuild settings over them."""
    clear_overlays()
    for profile in profiles:
        use_overlay(profile)
    reload_settings()
    reload_agent_settings()


def _score(predictions: list[Prediction], sweep: bool) -> IntentScore:
    """Scored at the routing threshold; a model row also at the sweep's.

    A router row has already applied the threshold (a turn it left to the LLM is
    UNKNOWN), so reading it at another one would mean nothing.
    """
    bakeoff = get_agent_settings().evals.intent_bakeoff
    provider = get_settings().llm.intent_classifier.provider
    threshold = get_agent_settings().intent_router.routing(provider).confidence_threshold
    levels = bakeoff.threshold_sweep if sweep else []
    return score(predictions, threshold, bakeoff.calibration_bins, levels)
