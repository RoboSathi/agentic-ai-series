"""The fixture world as types: authored pages and the search index over them."""

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class WorldPage(BaseModel):
    url: str
    title: str
    published: str
    body: str
    status: int = 200
    js_only: bool = False
    primary: bool = False
    mirror_of: str = ""
    hostile: str = ""


class WorldQuery(BaseModel):
    query: str
    keywords: list[str]
    urls: list[str]
    answer: str = ""


class WorldTopic(BaseModel):
    """What the demo world can answer, as the greeting lists it."""

    title: str
    example: str


class World(BaseModel):
    set_name: str
    version: str
    description: str
    # "Now" in this world: publication dates are fixed, so recency windows
    # are measured from here, not from the wall clock.
    as_of: datetime
    topics: list[WorldTopic] = Field(default_factory=list)
    # One question per mode that the demo world answers well.
    examples: dict[str, str] = Field(default_factory=dict)
    reader_url_template: str
    persona: "ModelPersona"
    pages: list[WorldPage]
    queries: list[WorldQuery]

    def page(self, url: str) -> WorldPage | None:
        return next((page for page in self.pages if page.url == url), None)

    def hostile_pages(self) -> list[WorldPage]:
        return [page for page in self.pages if page.hostile]


class ModelPersona(BaseModel):
    """How the reference model behaves: its naivety is the point of the ablations."""

    vague_answer_words: int
    absolute_words: list[str] = Field(default_factory=list)
    none_answers: list[str] = Field(default_factory=list)
    challenge: str
    fallback_question: str
    tokens_per_char: float
    exfiltration_marker: str
    override_marker: str
    persuaded_by: list[str] = Field(default_factory=list)


World.model_rebuild()


def load_world(path: Path) -> World:
    return World.model_validate(yaml.safe_load(path.read_text()))
