"""Type definitions for episodic memory."""

from pydantic import BaseModel


class Episode(BaseModel):
    """An episode in episodic memory."""

    id: str = ""
    description: str = ""
