"""`make model-check`: every model the system configures, inventoried once and probed once."""

from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.llm import probes
from agent_platform.llm.benchmark import BenchmarkResult, ChatModel, _summary, chat_models
from agent_platform.llm.inventory import ModelUse, grouped, merge, uses_in
from agent_platform.llm.local_probes import build_local_probes
from agent_platform.llm.probes import ChatProbe, GuardProbe, guard_probe
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


class TestInventory:
    def test_every_hosted_model_is_listed_with_what_uses_it(self) -> None:
        config = get_settings()
        uses = uses_in(config, "platform")
        models = {(use.kind, use.model) for use in uses}
        assert ("chat", config.llm.provider_settings().model) in models
        assert ("chat", config.judge.model) in models
        for fallback in config.llm.chain.fallbacks:
            assert ("chat", fallback.model) in models
        words = config.llm.check.uses
        assert {use.use for use in uses} >= {words["chat"], words["judge"]}
        assert all(use.layer == "platform" for use in uses)

    def test_a_fallback_alone_is_not_required(self) -> None:
        def use(model: str, primary: bool, layer: str = "platform") -> ModelUse:
            kind = "chat"
            return ModelUse(
                kind=kind, provider="nim", model=model, use="u", layer=layer, primary=primary
            )

        entries = {
            entry.model: entry
            for entry in merge([use("a", True), use("b", False), use("b", True, "csa")])
        }
        assert entries["a"].required and entries["b"].required
        assert not merge([use("c", False)])[0].required

    def test_a_model_named_twice_is_one_entry_with_every_layer(self) -> None:
        assert grouped([("chat", "platform"), ("chat", "csa"), ("judge", "platform")]) == [
            "chat (platform, csa)",
            "judge (platform)",
        ]

    def test_a_disabled_guard_is_not_listed(self, restore_settings: None) -> None:
        config = get_settings()
        config.security.guards.jailbreak.enabled = False
        kinds = {use.model for use in uses_in(config, "platform") if use.kind == "guard"}
        assert config.security.guards.jailbreak.model not in kinds

    def test_a_guard_probe_targets_the_guard_that_names_the_model(self) -> None:
        config = get_settings()
        model = config.security.guards.content_safety.model
        probe = guard_probe(config, model, required=True)
        assert isinstance(probe, GuardProbe) and probe._model == model
        with pytest.raises(KeyError):
            guard_probe(config, "no/such-model", required=True)


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
        result = await ChatProbe("judge", "nim", "m", True, get_settings()).probe()
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
