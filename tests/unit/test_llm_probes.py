"""`make llm-check`: one real, minimal request per hosted endpoint."""

from typing import Any

import pytest

from agentic_commons.config import get_settings
from agentic_commons.llm import probes
from agentic_commons.llm.probes import (
    ChatProbe,
    GuardProbe,
    SkippedProbe,
    build_probes,
    local_models,
)
from agentic_commons.security.guard_types import GuardVerdict
from tests.conftest import ScriptedLLM


class _FakeGuard:
    def __init__(self, verdict: GuardVerdict) -> None:
        self.verdict = verdict
        self.seen: list[str] = []

    async def classify(self, text: str) -> GuardVerdict:
        self.seen.append(text)
        return self.verdict


class _Exploding(ScriptedLLM):
    async def complete(self, request: Any) -> Any:
        error = RuntimeError("Not found for account")
        error.status_code = 404  # type: ignore[attr-defined]
        raise error


class TestBuildProbes:
    def test_one_probe_per_configured_hosted_endpoint(self) -> None:
        names = [probe.name for probe in build_probes()]
        assert names == ["agent_chat", "eval_judge", "embeddings", "content_safety", "jailbreak"]

    def test_the_judge_probe_targets_the_judge_model_not_the_agents(self) -> None:
        judge = next(probe for probe in build_probes() if probe.name == "eval_judge")
        assert isinstance(judge, ChatProbe)
        assert judge._model == get_settings().evals.judge.model

    def test_a_disabled_guard_is_skipped(self, restore_settings: None) -> None:
        get_settings().security.guards.jailbreak.enabled = False
        jailbreak = next(probe for probe in build_probes() if probe.name == "jailbreak")
        assert isinstance(jailbreak, SkippedProbe)


class TestChatProbe:
    async def test_a_completion_is_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probes, "create_pinned_llm_client", lambda *_: ScriptedLLM())
        result = await ChatProbe("agent_chat", "nim", "m", True, get_settings()).probe()
        assert result.status == "ok"
        assert result.detail == "ok"

    async def test_an_unentitled_model_fails_with_its_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(probes, "create_pinned_llm_client", lambda *_: _Exploding())
        result = await ChatProbe("eval_judge", "nim", "m", True, get_settings()).probe()
        assert result.status == "failed"
        assert result.http_status == 404


class TestGuardProbe:
    async def test_an_available_verdict_is_ok(self) -> None:
        settings = get_settings().security.guards.content_safety
        fake = _FakeGuard(GuardVerdict(decision="allow"))
        result = await GuardProbe("cs", settings, True, lambda _: fake, "hi").probe()  # type: ignore[arg-type,return-value]
        assert result.status == "ok"
        assert fake.seen == ["hi"]

    async def test_an_unavailable_guard_fails(self) -> None:
        settings = get_settings().security.guards.content_safety
        fake = _FakeGuard(GuardVerdict(decision="unavailable", http_status=503, reason="down"))
        result = await GuardProbe("cs", settings, True, lambda _: fake, "hi").probe()  # type: ignore[arg-type,return-value]
        assert (result.status, result.http_status, result.detail) == ("failed", 503, "down")


def test_local_models_report_presence() -> None:
    names = {model.name for model in local_models()}
    assert "nli" in names
