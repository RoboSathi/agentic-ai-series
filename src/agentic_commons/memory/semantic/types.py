"""Type definitions for semantic memory."""

from pydantic import BaseModel


class Fact(BaseModel):
    """A fact in semantic memory."""

    id: str = ""
    content: str = ""
