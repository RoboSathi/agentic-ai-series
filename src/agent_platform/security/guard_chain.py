"""A hosted guard and its fallbacks, asked in turn (Chain of Responsibility).

A hosted classifier that does not answer — an outage, a timeout, its breaker open, the
account's rate limit — used to leave the fail-closed policy as the only answer, so an
unreachable model quarantined ordinary messages. The chain asks the next model instead,
and only when every one is unavailable is the verdict *unavailable*. A verdict a fallback
gave says so (`security.guards.fallback_reason`), so the card shows who actually decided.
"""

from collections.abc import Sequence
from typing import Protocol

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.security.guard_types import GuardVerdict

_UNAVAILABLE = "unavailable"


class TextGuard(Protocol):
    """Any guard that classifies one text (a jailbreak or a content-safety classifier)."""

    @property
    def model(self) -> str: ...

    async def classify(self, text: str) -> GuardVerdict: ...


class GuardChain:
    """The primary guard, then each fallback, until one answers."""

    def __init__(self, guards: Sequence[TextGuard]) -> None:
        self._app_config = get_settings()
        if not guards:
            raise ValueError("a guard chain needs at least one guard")
        self._guards = list(guards)
        self._logger = get_logger("security")

    @property
    def model(self) -> str:
        """The primary's model: the one the chain is configured to ask first."""
        return self._guards[0].model

    @property
    def models(self) -> list[str]:
        return [guard.model for guard in self._guards]

    async def classify(self, text: str) -> GuardVerdict:
        """The first verdict that is not *unavailable*; the last unavailable one otherwise."""
        verdict = await self._guards[0].classify(text)
        for guard in self._guards[1:]:
            if verdict.decision != _UNAVAILABLE:
                break
            self._logger.warning(
                "Guard unavailable; asking the next model",
                unavailable=verdict.model,
                next=guard.model,
                reason=verdict.reason,
            )
            verdict = self._answered_by_fallback(await guard.classify(text))
        return verdict

    def _answered_by_fallback(self, verdict: GuardVerdict) -> GuardVerdict:
        if verdict.decision == _UNAVAILABLE:
            return verdict
        wording = self._app_config.security.guards.fallback_reason
        reason = wording.format(reason=verdict.reason, model=verdict.model, primary=self.model)
        return verdict.model_copy(update={"reason": reason})
