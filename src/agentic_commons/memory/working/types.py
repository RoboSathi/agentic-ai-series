"""Type definitions for working memory."""

from pydantic import BaseModel


class Context(BaseModel):
    """Working memory context."""

    session_id: str = ""
    data: dict = {}
