"""The CSA bake-off runner end to end, with a scripted classifier: no model is downloaded."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_platform.evals.intent import render_report
from agent_platform.intent import IntentCandidate
from customer_support_agent.config import get_agent_settings
from customer_support_agent.config.settings import reload_agent_settings
from customer_support_agent.evals.intent import bakeoff


class AlwaysUnknown:
    model_name = "scripted"
    score_type = "model_softmax"
    is_loaded = True

    async def ensure_loaded(self) -> None:
        return None

    def ensure_loaded_sync(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        return [IntentCandidate(intent="fallback", score=0.9)]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        "cases:\n"
        "  - {text: 'refund order 10842', intent: REQUEST_REFUND}\n"
        "  - {text: 'tell me a joke', intent: UNKNOWN}\n"
        "  - {text: 'scrap order 10130', intent: CANCEL_ORDER}\n"
    )
    monkeypatch.setattr(bakeoff, "create_intent_classifier", lambda **_: AlwaysUnknown())

    def settings_with_tmp_paths() -> None:
        section = get_agent_settings().evals.intent_bakeoff
        section.cases_path = str(cases)
        section.contenders = {"scripted": []}

    real_apply = bakeoff._apply

    def apply(profile: Any) -> None:
        real_apply(profile)
        settings_with_tmp_paths()

    monkeypatch.setattr(bakeoff, "_apply", apply)
    settings_with_tmp_paths()
    yield tmp_path
    reload_agent_settings()


@pytest.mark.asyncio
async def test_scores_both_modes_and_renders_one_report(scripted: Path) -> None:
    result = await bakeoff.run()

    assert set(result.scores) == {"scripted · model", "scripted · router"}
    report = render_report(result.title, result.scores, result.misses, z=1.96)
    assert "scripted · model" in report and "scripted · router" in report


@pytest.mark.asyncio
async def test_rules_score_in_the_router_row_but_not_the_model_row(scripted: Path) -> None:
    result = await bakeoff.run()

    assert result.scores["scripted · model"].correct == 1
    assert result.scores["scripted · router"].correct == 2


def test_loads_the_shipped_cases() -> None:
    cases = bakeoff.load_cases(get_agent_settings().evals.intent_bakeoff.resolved_cases_path)

    assert len(cases) == 50
    assert {case.intent for case in cases} == {i.name for i in get_agent_settings().intents}


def test_a_contender_applies_each_of_its_profiles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[str] = []
    monkeypatch.setattr(bakeoff, "use_overlay", applied.append)

    bakeoff._apply(["first.yaml", "second.yaml"])
    bakeoff._apply([])

    assert applied == ["first.yaml", "second.yaml"]
