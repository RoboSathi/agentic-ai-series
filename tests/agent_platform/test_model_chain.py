"""Backups and hedging: one platform implementation every model call goes through."""

import asyncio
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import ModelChainSettings, ModelRef, RoleModelSettings
from agent_platform.llm.cascade import CascadingLLMClient
from agent_platform.llm.chain import ModelChainBuilder
from agent_platform.llm.client import LLMClient
from agent_platform.llm.resilient import ResilientLLMClient
from agent_platform.llm.roles import RoleModelRouter
from agent_platform.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agent_platform.resilience.hedge import HedgedCall

REQUEST = CompletionRequest(messages=[])


class _Model(LLMClient):
    """A model that answers after `delay` seconds, or raises."""

    def __init__(self, name: str, delay: float = 0.0, fails: bool = False) -> None:
        self.name, self.delay, self.fails, self.calls = name, delay, fails, 0

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fails:
            raise RuntimeError(f"{self.name} down")
        return CompletionResult(content=self.name, model=self.name)

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)

    @property
    def model(self) -> str:
        return self.name

    @property
    def tier(self) -> str:
        return self.name


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []

    def backup_started(self, label: str, primary: str, backup: str, reason: str) -> None:
        self.events.append(("started", label, primary, backup, reason))

    def backup_answered(self, label: str, primary: str, backup: str) -> None:
        self.events.append(("answered", label, primary, backup))


def _cascade(
    *models: _Model, hedge: float | None = None, observer: Any = None
) -> CascadingLLMClient:
    settings = get_settings().llm.cascade.model_copy(update={"escalate_on_finish_reasons": []})
    return CascadingLLMClient(
        list(models), settings, hedge_after_seconds=hedge, observer=observer, label="role"
    )


# -- HedgedCall ---------------------------------------------------------------------------


async def _value(value: str, delay: float = 0.0, fails: bool = False) -> str:
    await asyncio.sleep(delay)
    if fails:
        raise RuntimeError(value)
    return value


async def test_the_primary_answers_and_no_backup_starts() -> None:
    index, value = await HedgedCall(0.5).run([lambda: _value("a"), lambda: _value("b")])
    assert (index, value) == (0, "a")


async def test_a_slow_primary_is_overtaken_after_the_hedge_delay() -> None:
    index, value = await HedgedCall(0.05).run([lambda: _value("a", 1.0), lambda: _value("b")])
    assert (index, value) == (1, "b")


async def test_a_failed_primary_starts_the_backup_at_once_without_hedging() -> None:
    index, value = await HedgedCall(None).run(
        [lambda: _value("a", fails=True), lambda: _value("b")]
    )
    assert (index, value) == (1, "b")


async def test_every_option_failing_raises_the_last_error() -> None:
    with pytest.raises(RuntimeError, match="b"):
        await HedgedCall(None).run(
            [lambda: _value("a", fails=True), lambda: _value("b", fails=True)]
        )


async def test_a_failure_that_must_not_move_on_is_raised_at_once() -> None:
    call = HedgedCall(None, moves_on=lambda error: False)
    with pytest.raises(RuntimeError, match="a"):
        await call.run([lambda: _value("a", fails=True), lambda: _value("b")])


# -- the cascade as backups ----------------------------------------------------------------


async def test_a_slow_model_is_answered_by_its_backup_and_the_observer_is_told() -> None:
    observer = _Observer()
    primary, backup = _Model("big", delay=1.0), _Model("small")
    result = await _cascade(primary, backup, hedge=0.05, observer=observer).complete(REQUEST)
    assert result.model == "small"
    assert observer.events == [
        ("started", "role", "big", "small", "slow"),
        ("answered", "role", "big", "small"),
    ]


async def test_a_failed_model_falls_back_in_order() -> None:
    first, second, third = _Model("a", fails=True), _Model("b", fails=True), _Model("c")
    result = await _cascade(first, second, third).complete(REQUEST)
    assert result.model == "c" and (first.calls, second.calls, third.calls) == (1, 1, 1)


# -- the builder and the router ------------------------------------------------------------


def test_a_role_with_backups_is_a_cascade_of_protected_models() -> None:
    made: list[str] = []

    def factory(provider: str, model: str) -> LLMClient:
        made.append(model)
        return _Model(model)

    spec = RoleModelSettings(
        provider="nim", model="big", fallbacks=[ModelRef(provider="nim", model="small")]
    )
    router = RoleModelRouter({"writer": spec, "judge": spec}, factory=factory)
    client = router.client_for("writer")
    assert isinstance(client, CascadingLLMClient)
    assert all(isinstance(tier, ResilientLLMClient) for tier in client._tiers)
    router.client_for("judge")
    assert made == ["big", "small"]  # one adapter per distinct model, shared by both roles


def test_a_model_without_backups_is_just_protected() -> None:
    builder = ModelChainBuilder(lambda provider, model: _Model(model))
    client = builder.build(ModelRef(provider="nim", model="m"), ModelChainSettings(), label="x")
    assert isinstance(client, ResilientLLMClient)


async def test_the_result_says_how_many_attempts_it_took() -> None:
    builder = ModelChainBuilder(lambda provider, model: _Model(model))
    client = builder.build(ModelRef(provider="nim", model="m"), ModelChainSettings(), label="x")
    assert (await client.complete(REQUEST)).attempts == 1
