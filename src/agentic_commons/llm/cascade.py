"""Model cascading — cheapest sufficient tier first (E44, LLD §3).

The same shape as the NOLLM scoring cascade: run the cheap tier, escalate
only when its answer doesn't resolve the case. The escalation rule is a
Strategy so "when is cheap good enough?" can change without touching the
client.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.config.settings import LLMCascadeSettings
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agentic_commons.logging import get_logger


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


class CascadingLLMClient(LLMClient):
    """Runs an ordered list of tiers, escalating only when needed."""

    def __init__(
        self,
        tiers: list[LLMClient],
        settings: LLMCascadeSettings,
        policy: EscalationPolicy | None = None,
    ) -> None:
        if not tiers:
            raise ValueError("A cascade needs at least one tier")
        self._tiers = tiers
        self._settings = settings
        self._policy = policy or ConfiguredEscalationPolicy(settings)
        self._logger = get_logger("llm")

    @property
    def tier(self) -> str:
        return "+".join(tier.tier for tier in self._tiers)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return the first tier's result that the policy accepts.

        The returned result carries the tier that actually resolved it, so
        cost is attributed to the tier that did the work (E44).
        """
        last: CompletionResult | None = None
        for index, tier in enumerate(self._tiers):
            is_last = index == len(self._tiers) - 1
            try:
                result = await tier.complete(request)
            except Exception as exc:
                if not self._settings.escalate_on_error or is_last:
                    raise
                self._logger.warning(f"Tier {tier.tier} failed, escalating", error=str(exc))
                continue
            last = result
            if is_last or not self._policy.should_escalate(result):
                return result
            self._logger.info(
                "Escalating to the next tier",
                from_tier=tier.tier,
                finish_reason=result.finish_reason,
            )
        assert last is not None  # a non-empty cascade always produces a result
        return last

    async def health_check(self) -> HealthCheckResult:
        """Healthy only if every tier is reachable."""
        for tier in self._tiers:
            result = await tier.health_check()
            if not result.healthy:
                return result
        return HealthCheckResult(healthy=True)
