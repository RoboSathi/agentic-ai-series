"""Type definitions for observability module."""

from pydantic import BaseModel


class SpanHandle(BaseModel):
    """Handle to a tracing span."""

    id: str = ""
