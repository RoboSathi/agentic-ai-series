"""Type definitions for LLM module."""

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Response from LLM generation."""

    content: str = ""
    model: str = ""
    tokens_used: int = 0
