"""`make llm-check`: one real, minimal request per hosted endpoint."""

from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.llm import probes
from agent_platform.llm.benchmark import BenchmarkResult, ChatModel, _summary, chat_models
from agent_platform.llm.local_probes import build_local_probes
from agent_platform.llm.probes import (
    ChatProbe,
    GuardProbe,
    SkippedProbe,
    build_probes,
)
from agent_platform.security.guard_types import GuardVerdict
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


def test_every_configured_local_model_gets_a_probe() -> None:
    names = {probe.name for probe in build_local_probes()}
    assert {"nli", "dialogue_summarizer"} <= names


def test_each_distinct_role_model_is_probed_under_its_role_names() -> None:
    settings = get_settings().model_copy(deep=True)
    role = settings.llm.provider_settings()
    settings.llm.roles = {
        "planner": _role(role.model),
        "writer": _role(role.model),
        "judge": _role("another/model"),
    }
    names = {probe.name for probe in build_probes(settings)}
    assert "roles: planner, writer" in names and "roles: judge" in names


def test_chat_models_are_listed_once_with_every_use() -> None:
    settings = get_settings().model_copy(deep=True)
    settings.llm.roles = {"answerer": _role(settings.llm.provider_settings().model)}
    models = {m.model: m.uses for m in chat_models(settings)}
    assert "role:answerer" in models[settings.llm.provider_settings().model]


def test_the_benchmark_summary_reports_median_and_throughput() -> None:
    target = ChatModel(provider="nim", model="m", uses=["role:judge"])
    result = _summary(target, 3, [1.0, 2.0, 4.0], tokens=70, errors=[])
    assert isinstance(result, BenchmarkResult)
    assert (result.p50_seconds, result.p95_seconds) == (2.0, 4.0)
    assert result.completion_tokens_per_second == 10.0
    failed = _summary(target, 2, [], tokens=0, errors=["timeout", "timeout"])
    assert (failed.failures, failed.detail, failed.p50_seconds) == (2, "timeout", 0.0)


def _role(model: str) -> Any:
    from agent_platform.config.settings import RoleModelSettings

    return RoleModelSettings(provider="nim", model=model)
