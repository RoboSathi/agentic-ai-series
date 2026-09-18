"""One security call site for the running product (LLD §12, Facade).

`customer_support_agent` should not wire four security components by
hand and get the ordering subtly wrong. The order matters: minimize
before the call leaves, separate before the model reads, redact before
anything persists.
"""

from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument
from agentic_commons.security.auth.auth import Authorizer
from agentic_commons.security.auth.types import AccessDecision, Principal
from agentic_commons.security.content_moderation.moderator import (
    ContentModerator,
    TermContentModerator,
)
from agentic_commons.security.content_moderation.types import ModerationVerdict
from agentic_commons.security.pii.detector import PIIFilterImpl
from agentic_commons.security.pii.types import RedactionMode
from agentic_commons.security.prompt_injection.detector import PromptInjectionGuard
from agentic_commons.security.prompt_injection.types import InjectionAssessment
from agentic_commons.tools.types import ToolDefinition


@instrument("security")
class SecuritySuite:
    """Auth, PII, injection defense and moderation behind one object."""

    def __init__(
        self,
        authorizer: Authorizer | None = None,
        pii: PIIFilterImpl | None = None,
        injection: PromptInjectionGuard | None = None,
        moderator: ContentModerator | None = None,
    ) -> None:
        self.authorizer = authorizer or Authorizer()
        self.pii = pii or PIIFilterImpl()
        self.injection = injection or PromptInjectionGuard()
        self.moderator = moderator or TermContentModerator()
        self._enabled = get_settings().security.enabled
        self._logger = get_logger("security")

    def authorize(self, principal: Principal, action: str) -> AccessDecision:
        """Whether this principal may perform this action."""
        return self.authorizer.authorize(principal, action)

    def prepare_untrusted(
        self, instructions: str, data: str, tools: list[ToolDefinition]
    ) -> InjectionAssessment:
        """Assess untrusted content and narrow the toolset accordingly."""
        return self.injection.assess(instructions, data, tools)

    def redact(self, text: str, mode: RedactionMode | None = None) -> str:
        """Remove personal data before the text is logged or persisted."""
        return self.pii.redact(text, mode)

    def minimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Strip fields a tool call is not entitled to carry."""
        return self.pii.minimize(payload)

    def moderate(self, text: str) -> ModerationVerdict:
        """Advisory content check — flags, never blocks."""
        return self.moderator.moderate(text)

    @property
    def enabled(self) -> bool:
        """Whether the security package is active at all."""
        return self._enabled
