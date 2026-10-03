"""Live search answers kept on disk for a while (a TTL cache).

Metered search is the scarcest resource an agent spends: re-asking the
same question, or re-running while developing, should cost nothing. Only
searches are cached — pages are fetched fresh, because a page is evidence
and must be the text this run actually retrieved.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

from agent_platform.web.types import SearchRequest, SearchResponse

Clock = Callable[[], float]
_SECONDS_PER_HOUR = 3600.0
_SUFFIX = ".json"


class SearchCache:
    """`get`/`put` search responses by request key; entries expire after `ttl_hours`."""

    def __init__(self, folder: Path, ttl_hours: float, clock: Clock = time.time) -> None:
        self._folder = folder
        self._ttl = ttl_hours * _SECONDS_PER_HOUR
        self._clock = clock

    def get(self, request: SearchRequest) -> SearchResponse | None:
        path = self._path(request)
        if self._ttl <= 0 or not path.exists():
            return None
        stored = json.loads(path.read_text())
        if self._clock() - float(stored["stored_at"]) > self._ttl:
            return None
        return SearchResponse.model_validate(stored["response"])

    def put(self, response: SearchResponse) -> None:
        if self._ttl <= 0:
            return
        self._folder.mkdir(parents=True, exist_ok=True)
        entry = {"stored_at": self._clock(), "response": response.model_dump(mode="json")}
        self._path(response.request).write_text(json.dumps(entry))

    def _path(self, request: SearchRequest) -> Path:
        return self._folder / f"{request.key}{_SUFFIX}"
