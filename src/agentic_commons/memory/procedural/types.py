"""Type definitions for procedural memory."""

from pydantic import BaseModel


class Skill(BaseModel):
    """A skill in procedural memory."""

    name: str = ""
    implementation: str = ""
