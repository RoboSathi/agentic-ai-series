"""Tier-2 semantic advisory (E12, LLD §11).

Tier 1 is deterministic and blocking. Tier 2 is semantic and advisory,
and the separation is absolute: a model-based check may add context to a
decision and may raise an escalation, but it can never be the reason an
action was permitted or refused.

The reason is not distrust of models in general. It is that a semantic
check can be argued with — in natural language, by the same untrusted
input it is inspecting. A control that can be talked out of its verdict
is not a control, so it is given no verdict to be talked out of: the
interface returns `AdvisoryFlag`, which has no field that stops anything.
"""

from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.gatekeeper.types import AdvisoryFlag
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger
from agent_platform.resilience import external_api_error_fields
from agent_platform.tools.types import ToolCall


@runtime_checkable
class Tier2Advisor(Protocol):
    """Flags a call for attention.

    Note the return type: `AdvisoryFlag`, never `AuthorityDecision`. The
    non-blocking constraint is enforced by this signature.
    """

    async def flag(self, call: ToolCall) -> AdvisoryFlag: ...


class SemanticTier2Advisor:
    """LLM-backed advisory review, structurally unable to block."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client
        tier2 = get_settings().gatekeeper.tier2
        self._enabled = tier2.enabled
        # Configured, not literal: "unusual for a *what*?" is the whole
        # question, and the platform does not know what it is running.
        self._prompt = tier2.prompt
        self._logger = get_logger("gatekeeper")

    async def flag(self, call: ToolCall) -> AdvisoryFlag:
        """Ask for an advisory note; a failure here never affects the verdict.

        If the model is down, slow, or returns nonsense, the result is an
        unflagged advisory — because Tier 2 being unavailable must not
        change what Tier 1 decides.
        """
        if not self._enabled or self._client is None:
            return AdvisoryFlag(reason="tier-2 advisory disabled")
        try:
            result = await self._client.complete(
                CompletionRequest(
                    messages=[
                        Message(
                            role="user",
                            content=self._prompt.format(action=call.action, params=call.params),
                        )
                    ]
                )
            )
        except Exception as exc:
            self._logger.warning(
                "Tier-2 advisor unavailable",
                error=str(exc),
                **(external_api_error_fields(exc) or {}),
            )
            return AdvisoryFlag(reason="advisor unavailable; decision unaffected")
        return AdvisoryFlag(
            flagged=bool(result.content.strip()),
            reason=result.content.strip(),
            confidence=0.0,
        )


class NullTier2Advisor:
    """The advisor used when Tier 2 is off — always silent."""

    async def flag(self, call: ToolCall) -> AdvisoryFlag:
        """Never flags. Tier 1 decides alone."""
        return AdvisoryFlag(reason="no tier-2 advisor configured")
