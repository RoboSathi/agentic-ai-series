"""Advisory content moderation (LLD §12.4).

Held to the Gatekeeper's Tier-2 constraint: this classifier may flag and
escalate, never block. The constraint is enforced by the interface, which
returns a `ModerationVerdict` and has no path to an authorization
decision — the same structural guarantee `SemanticTier2Advisor` relies on.
"""

import re
from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged
from agentic_commons.security.content_moderation.types import ModerationVerdict


@runtime_checkable
class ContentModerator(Protocol):
    """Flags content for human attention. Advisory only."""

    def moderate(self, text: str) -> ModerationVerdict: ...


class TermContentModerator:
    """Term-list moderation, matched on word boundaries.

    Substring matching is what produces the classic false positives —
    "assist" flagged for containing a slur fragment, "Scunthorpe" for the
    obvious reason. Word boundaries cost nothing and remove that class of
    error entirely.
    """

    def __init__(self) -> None:
        settings = get_settings().security.content_moderation
        self._terms = [term.casefold() for term in settings.blocked_terms]
        self._pattern = (
            re.compile(r"\b(" + "|".join(re.escape(term) for term in self._terms) + r")\b")
            if self._terms
            else None
        )
        self._logger = get_logger("content_moderation")

    @logged("content_moderation")
    def moderate(self, text: str) -> ModerationVerdict:
        """Flag matched terms; always advisory, never blocking."""
        if self._pattern is None:
            return ModerationVerdict(reason="no blocked terms are configured")
        matched = sorted(set(self._pattern.findall(text.casefold())))
        if not matched:
            return ModerationVerdict(reason="no flagged terms found")
        self._logger.warning("Content flagged for review", terms=matched)
        return ModerationVerdict(
            flagged=True,
            categories=["blocked_term"],
            matched_terms=matched,
            reason=f"{len(matched)} flagged term(s) present",
        )
