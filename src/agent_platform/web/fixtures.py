"""Recorded web traffic, so a run can be replayed without the network.

Fixtures are keyed by a hash of the *normalised* request and grouped into
named, versioned sets, each with the date it was recorded. Replaying a
set reproduces a run exactly; a request the set never saw is a loud miss.

    <root>/<set>/manifest.json            {"set": ..., "version": ..., "recorded_at": ...}
    <root>/<set>/search/<key>.json        SearchResponse
    <root>/<set>/fetch/<key>.json         FetchedPage
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.observability.types import utc_now
from agent_platform.web.types import FetchedPage, SearchResponse

FixtureKind = Literal["search", "fetch"]
_MANIFEST = "manifest.json"


class FixtureManifest(BaseModel):
    """What a fixture set is and when it was recorded."""

    set_name: str
    version: str
    recorded_at: datetime = Field(default_factory=utc_now)
    description: str = ""


class FixtureStore:
    """Reads and writes one named fixture set."""

    def __init__(self, root: Path, set_name: str) -> None:
        self._folder = root / set_name
        self._set_name = set_name

    @property
    def folder(self) -> Path:
        return self._folder

    def manifest(self) -> FixtureManifest | None:
        path = self._folder / _MANIFEST
        return FixtureManifest.model_validate_json(path.read_text()) if path.exists() else None

    def search(self, key: str) -> SearchResponse | None:
        raw = self._read("search", key)
        return SearchResponse.model_validate_json(raw) if raw is not None else None

    def fetch(self, key: str) -> FetchedPage | None:
        raw = self._read("fetch", key)
        return FetchedPage.model_validate_json(raw) if raw is not None else None

    def put_search(self, key: str, response: SearchResponse) -> Path:
        return self._write("search", key, response.model_dump_json(indent=2))

    def put_fetch(self, key: str, page: FetchedPage) -> Path:
        return self._write("fetch", key, page.model_dump_json(indent=2))

    def keys(self, kind: FixtureKind) -> list[str]:
        folder = self._folder / kind
        return sorted(path.stem for path in folder.glob("*.json")) if folder.exists() else []

    def _read(self, kind: FixtureKind, key: str) -> str | None:
        path = self._folder / kind / f"{key}.json"
        return path.read_text() if path.exists() else None

    def _write(self, kind: FixtureKind, key: str, body: str) -> Path:
        folder = self._folder / kind
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.json"
        json.loads(body)  # never write a fixture that cannot be read back
        path.write_text(body)
        return path
