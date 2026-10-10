"""The CSA bake-off runner end to end, with a scripted classifier: no model is downloaded."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

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
        section.report_dir = str(tmp_path / "reports")
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
async def test_writes_one_report_with_both_modes(scripted: Path) -> None:
    path = await bakeoff.run()

    report = path.read_text()
    assert path.parent == scripted / "reports"
    assert "scripted · model" in report
    assert "scripted · router" in report


@pytest.mark.asyncio
async def test_rules_score_in_the_router_row_but_not_the_model_row(scripted: Path) -> None:
    path = await bakeoff.run()

    summary = path.read_text().split("## Recall by intent")[0]
    model_row = next(line for line in summary.splitlines() if "scripted · model" in line)
    router_row = next(line for line in summary.splitlines() if "scripted · router" in line)
    assert "1/3" in model_row
    assert "2/3" in router_row


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
