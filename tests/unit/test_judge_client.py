"""Every judge grades with the configured judge model, behind its configured backups."""

from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.evals import judge_client
from tests.conftest import ScriptedLLM


def test_the_judge_is_its_model_then_its_backups(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, str]] = []

    def pinned(provider: Any, model: str, *_: Any) -> ScriptedLLM:
        built.append((provider, model))
        return ScriptedLLM()

    monkeypatch.setattr(judge_client, "create_pinned_llm_client", pinned)
    judge_client.create_judge_client()
    judge = get_settings().evals.judge
    assert built == [
        (judge.provider, judge.model),
        *((backup.provider, backup.model) for backup in judge.fallbacks),
    ]
    assert judge.fallbacks, "a stalled judge call has somewhere to go"
