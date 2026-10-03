"""Model cascading — cheapest sufficient tier first (E44, LLD §3), and backup models.

One ordered list of models serves two needs. *Quality*: run the cheap tier and escalate only
when its answer doesn't resolve the case — the escalation rule is a Strategy so "when is cheap
good enough?" can change without touching the client. *Availability*: when a tier fails, or
(with `hedge_after_seconds`) has not answered in time, the next one is asked — through the
platform's `HedgedCall`, so a slow model costs its hedge delay, not its whole timeout.
A role's backup models are this cascade with the configured escalation rules.
"""

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from agent_platform.config.settings import LLMCascadeSettings
from agent_platform.llm.client import LLMClient
from agent_platform.llm.observer import ModelCallObserver, NullModelObserver
from agent_platform.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agent_platform.logging import get_logger
from agent_platform.resilience.hedge import HedgedCall, StartReason


@runtime_checkable
class EscalationPolicy(Protocol):
    """Decides whether a tier's answer is good enough to return."""

    def should_escalate(self, result: CompletionResult) -> bool: ...


class ConfiguredEscalationPolicy:
    """Escalates on the signals named in config (empty answer, truncation…)."""

    def __init__(self, settings: LLMCascadeSettings) -> None:
        self._settings = settings

    def should_escalate(self, result: CompletionResult) -> bool:
        """True when this tier's result shouldn't be trusted as final."""
        if self._settings.escalate_on_empty_content and not (result.content or result.tool_calls):
            return True
        return result.finish_reason in self._settings.escalate_on_finish_reasons


# Why a backup started when the model before it answered, but not well enough.
WEAK = "weak"


class _EscalationError(Exception):
    """A tier answered, but not well enough: move on, and keep the answer in case none is."""

    def __init__(self, result: CompletionResult) -> None:
        super().__init__("answer escalated")
        self.result = result


class CascadingLLMClient(LLMClient):
    """Runs an ordered list of tiers: escalating on a weak answer, moving on from a failed or
    slow one."""

    def __init__(
        self,
        tiers: list[LLMClient],
        settings: LLMCascadeSettings,
        policy: EscalationPolicy | None = None,
        hedge_after_seconds: float | None = None,
        observer: ModelCallObserver | None = None,
        label: str = "",
    ) -> None:
        if not tiers:
            raise ValueError("A cascade needs at least one tier")
        self._tiers = tiers
        self._settings = settings
        self._policy = policy or ConfiguredEscalationPolicy(settings)
        self._hedge = HedgedCall(hedge_after_seconds, moves_on=self._moves_on)
        self._observer = observer or NullModelObserver()
        self._label = label
        self._logger = get_logger("llm")

    @property
    def tier(self) -> str:
        return "+".join(tier.tier for tier in self._tiers)

    @property
    def model(self) -> str:
        return self._tiers[0].model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return the first tier's result that the policy accepts.

        The returned result carries the tier that actually resolved it, so
        cost is attributed to the tier that did the work (E44).
        """
        options = [self._option(index, request) for index in range(len(self._tiers))]
        try:
            _, result = await self._hedge.run(options, _Listener(self))
        except _EscalationError as weak:
            return weak.result
        return result

    def _option(
        self, index: int, request: CompletionRequest
    ) -> Callable[[], Awaitable[CompletionResult]]:
        tier, last = self._tiers[index], index == len(self._tiers) - 1

        async def attempt() -> CompletionResult:
            result = await tier.complete(request)
            if not last and self._policy.should_escalate(result):
                self._logger.info(
                    "Escalating to the next tier",
                    from_tier=tier.tier,
                    finish_reason=result.finish_reason,
                )
                raise _EscalationError(result)
            return result

        return attempt

    def _moves_on(self, error: BaseException) -> bool:
        """A weak answer always moves on; a failure only when the settings say so."""
        return isinstance(error, _EscalationError) or self._settings.escalate_on_error

    def _started(self, index: int, reason: StartReason, error: BaseException | None) -> None:
        primary, backup = self._tiers[0].model, self._tiers[index].model
        # A weak answer (empty, cut off) is not a failed call: say which it was.
        cause = WEAK if isinstance(error, _EscalationError) else reason
        self._logger.warning(
            "Backup model started", label=self._label, primary=primary, backup=backup, reason=cause
        )
        self._observer.backup_started(self._label, primary, backup, cause)

    def _answered(self, index: int) -> None:
        if index:
            self._observer.backup_answered(
                self._label, self._tiers[0].model, self._tiers[index].model
            )

    async def health_check(self) -> HealthCheckResult:
        """Healthy only if every tier is reachable."""
        for tier in self._tiers:
            result = await tier.health_check()
            if not result.healthy:
                return result
        return HealthCheckResult(healthy=True)


class _Listener:
    """`HedgeListener` for a cascade: index events to its observer, in model names."""

    def __init__(self, cascade: CascadingLLMClient) -> None:
        self._cascade = cascade

    def started(self, index: int, reason: StartReason, error: BaseException | None) -> None:
        self._cascade._started(index, reason, error)

    def answered(self, index: int) -> None:
        self._cascade._answered(index)
