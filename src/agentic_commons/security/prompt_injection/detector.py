"""Prompt-injection defense, layers 1-3 (E30/E35, LLD §12.3).

Three layers live here: instruction/data separation, suspicious-phrase
detection, and tool scoping. Layers 4-6 — the Gatekeeper, anomaly
detection, and the void/voidable/ratifiable model — are deliberately not
reimplemented here; they are `gatekeeper` and `evals`.

That split *is* the thesis. The strongest layer was never a detector. A
detector that misses one phrasing fails open; an authorization boundary
that never grants the authority in the first place has nothing to miss.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged
from agentic_commons.security.prompt_injection.types import (
    InjectionAssessment,
    SeparatedContext,
)
from agentic_commons.tools.types import SurfaceClass, ToolDefinition

# Surfaces a request touched by untrusted content may never reach, whatever
# the content asks for. This is structure, not a tunable threshold.
_UNTRUSTED_FORBIDDEN_SURFACES: frozenset[SurfaceClass] = frozenset({"locked", "human_controlled"})


@runtime_checkable
class InjectionGuard(Protocol):
    """Separates trusted instruction from untrusted data, and scopes tools."""

    def separate(self, instructions: str, data: str) -> SeparatedContext: ...

    def scope_tools(
        self, context: SeparatedContext, tools: list[ToolDefinition]
    ) -> list[ToolDefinition]: ...


class InstructionDataSeparator:
    """Layer 1-2 — delimit untrusted data and flag injection phrasing.

    Delimiting does not make the data safe; it makes its provenance
    legible to every layer downstream, including the Gatekeeper's
    authority-rank check where `tool_observations` outranks nothing.
    """

    def __init__(self) -> None:
        settings = get_settings().security.prompt_injection
        self._open = settings.data_open_tag
        self._close = settings.data_close_tag
        self._phrases = [phrase.casefold() for phrase in settings.suspicious_phrases]
        self._logger = get_logger("security")

    @logged("security")
    def separate(self, instructions: str, data: str) -> SeparatedContext:
        """Wrap untrusted data in its delimiters and scan it for injections."""
        return SeparatedContext(
            instructions=instructions,
            data=data,
            delimited_data=f"{self._open}\n{self._neutralize(data)}\n{self._close}",
            suspicious_phrases=self.find_phrases(data),
        )

    def find_phrases(self, text: str) -> list[str]:
        """Known injection phrasings present in the text."""
        lowered = text.casefold()
        return [phrase for phrase in self._phrases if phrase in lowered]

    def _neutralize(self, data: str) -> str:
        """Strip the closing delimiter so data cannot escape its own envelope.

        Without this, untrusted content that contains the closing tag
        ends the data block early and the rest is read as instruction —
        the delimiter equivalent of SQL injection.
        """
        return data.replace(self._close, "")


class ToolScopeLimiter:
    """Layer 3 — narrow the toolset once untrusted content is in play.

    Scoping is an authority decision made *before* the model sees the
    content, so no phrasing in that content can widen it. The model
    cannot call a tool it was never offered.
    """

    def __init__(self) -> None:
        self._enabled = get_settings().security.prompt_injection.enabled
        self._logger = get_logger("security")

    @logged("security")
    def scope_tools(
        self, context: SeparatedContext, tools: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """The tools still on offer given what the untrusted data contained."""
        if not self._enabled:
            return tools
        scoped = [tool for tool in tools if tool.surface not in _UNTRUSTED_FORBIDDEN_SURFACES]
        if context.is_suspicious:
            scoped = [tool for tool in scoped if tool.reversible]
            self._logger.warning(
                "Injection phrasing detected; restricting to reversible tools",
                phrases=context.suspicious_phrases,
                remaining=len(scoped),
            )
        return scoped


class PromptInjectionGuard:
    """Layers 1-3 behind one call site (Facade)."""

    def __init__(
        self,
        separator: InstructionDataSeparator | None = None,
        limiter: ToolScopeLimiter | None = None,
    ) -> None:
        self._separator = separator or InstructionDataSeparator()
        self._limiter = limiter or ToolScopeLimiter()
        self._enabled = get_settings().security.prompt_injection.enabled

    def separate(self, instructions: str, data: str) -> SeparatedContext:
        """Delimit and scan untrusted data."""
        return self._separator.separate(instructions, data)

    def scope_tools(
        self, context: SeparatedContext, tools: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Narrow the toolset for this turn."""
        return self._limiter.scope_tools(context, tools)

    def assess(
        self, instructions: str, data: str, tools: list[ToolDefinition]
    ) -> InjectionAssessment:
        """Full assessment: what was found, and what remains callable."""
        if not self._enabled:
            return InjectionAssessment(
                suspicious=False,
                allowed_actions=[tool.qualified_name for tool in tools],
                reason="prompt injection defense disabled",
            )
        context = self.separate(instructions, data)
        allowed = self.scope_tools(context, tools)
        return InjectionAssessment(
            suspicious=context.is_suspicious,
            phrases=context.suspicious_phrases,
            allowed_actions=[tool.qualified_name for tool in allowed],
            reason=(
                "injection phrasing found in untrusted data"
                if context.is_suspicious
                else "no injection phrasing found"
            ),
        )
