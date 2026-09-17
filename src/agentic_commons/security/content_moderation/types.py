"""Type definitions for content moderation."""

from pydantic import BaseModel


class ModerationResult(BaseModel):
    """Result of content moderation."""

    flagged: bool = False
    categories: list[str] = []
