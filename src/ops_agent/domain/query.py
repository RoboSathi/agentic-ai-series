"""An answered query: the only evidence an ops answer's figures can rest on."""

from typing import Any

from pydantic import BaseModel, Field


class QueryRecord(BaseModel):
    """One answered query, as the analytics server returned it."""

    sql: str
    columns: list[str]
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    # The tool that answered it (qualified): which SQL the model wrote, which the catalogue did.
    tool: str = ""


__all__ = ["QueryRecord"]
