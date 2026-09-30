"""An in-process web that serves the fixture world, for recording it.

Answers exactly what the real services would, in their wire shapes:
Tavily's `POST /search` JSON, each page as HTML (a script shell for a
JavaScript-only page, a 404 for a dead link), and the reader service's
rendered text. Recording through it exercises the real adapters.
"""

import json
from urllib.parse import unquote

import httpx

from research_agent.fixture_world.world import World, WorldPage

_SHELL = (
    "<html><head><title>{title}</title><script src='app.js'></script></head>"
    "<body><div id='root'></div><noscript>Please enable JavaScript to view this page.</noscript></body></html>"
)


class MockWeb:
    """An `httpx.MockTransport` handler over the world."""

    def __init__(self, world: World, search_url: str) -> None:
        self._world = world
        self._search_url = search_url
        self._reader_prefix = world.reader_url_template.split("{url}")[0]
        self._queries = {" ".join(q.query.casefold().split()): q for q in world.queries}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "POST" and url == self._search_url:
            return self._search(json.loads(request.content))
        if url.startswith(self._reader_prefix):
            return self._reader(unquote(url[len(self._reader_prefix) :]))
        return self._page(url)

    def _search(self, body: dict[str, object]) -> httpx.Response:
        entry = self._queries.get(" ".join(str(body["query"]).casefold().split()))
        results = [self._hit(url, rank) for rank, url in enumerate(entry.urls if entry else [])]
        return httpx.Response(
            200,
            json={
                "query": body["query"],
                "answer": entry.answer if entry else "",
                "results": results,
                "response_time": 0.01,
                "request_id": "fixture",
            },
        )

    def _hit(self, url: str, rank: int) -> dict[str, object]:
        page = self._world.page(url)
        teaser = page.body.strip().splitlines()[0] if page and page.body.strip() else ""
        return {
            "url": url,
            "title": page.title if page else url,
            "content": teaser,
            "score": round(1.0 - rank / 10, 2),
            "published_date": page.published if page else "",
        }

    def _page(self, url: str) -> httpx.Response:
        page = self._world.page(url)
        if page is None:
            return httpx.Response(404, text="not found")
        if page.status >= 400:
            return httpx.Response(page.status, text="")
        return httpx.Response(
            200, text=_SHELL.format(title=page.title) if page.js_only else _html(page)
        )

    def _reader(self, url: str) -> httpx.Response:
        page = self._world.page(url)
        return httpx.Response(200, text=page.body) if page else httpx.Response(404, text="")


def _html(page: WorldPage) -> str:
    paragraphs = "".join(f"<p>{line}</p>" for line in page.body.strip().splitlines())
    return f"<html><head><title>{page.title}</title></head><body>{paragraphs}</body></html>"
