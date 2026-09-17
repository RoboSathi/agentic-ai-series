"""Content moderation stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.security.content_moderation.types import ModerationResult


class ContentModerator:
    """Content moderation interface stub."""

    @log_call("security", toggle_path="security.content_moderation.enabled")
    def moderate(self, text: str) -> ModerationResult:
        """Phase 0 stub — no real moderation yet."""
        return ModerationResult(flagged=False, categories=[])
