"""Fetching a page's text: plain HTTP first, a reader service when that fails.

Many pages render their content with JavaScript, so a plain GET returns a
shell ("please enable JavaScript") and no prose. Those are detected —
too little text, or a configured marker — and re-read through a reader
service that renders them (a Jina-Reader-style URL template). Both paths
satisfy the same `Fetcher` interface, so a caller never knows which ran.
"""

import re
from html.parser import HTMLParser
from typing import Protocol, runtime_checkable

import httpx

from agent_platform.config.settings import WebFetchSettings
from agent_platform.primitives.hashing import text_hash
from agent_platform.web.page_date import published_from_html
from agent_platform.web.types import FetchedPage, FetchVia

# `noscript` is kept: its "please enable JavaScript" text is how a shell page is recognised.
_SKIP_TAGS = frozenset({"script", "style", "template", "svg"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}
)
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


@runtime_checkable
class Fetcher(Protocol):
    """Retrieves one URL's readable text."""

    async def fetch(self, url: str) -> FetchedPage: ...


class _TextExtractor(HTMLParser):
    """Visible text only: scripts, styles and markup are dropped."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skipping = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skipping += 1
        self._in_title = tag == "title"
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skipping:
            self._skipping -= 1
        self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
        elif not self._skipping:
            self._parts.append(data)

    def text(self) -> str:
        joined = _SPACES.sub(" ", "".join(self._parts))
        return _BLANK_LINES.sub("\n\n", joined).strip()


def extract_text(html: str) -> tuple[str, str]:
    """(title, visible text) of an HTML document; plain text passes through."""
    if "<" not in html:
        return "", html.strip()
    parser = _TextExtractor()
    parser.feed(html)
    return parser.title, parser.text()


def page_from(
    url: str, status: int, body: str, via: FetchVia, settings: WebFetchSettings
) -> FetchedPage:
    """A `FetchedPage` whose hash is over exactly the text it carries."""
    title, text = extract_text(body)
    clipped = text[: settings.max_chars]
    return FetchedPage(
        url=url,
        status=status,
        text=clipped,
        page_hash=text_hash(clipped),
        via=via,
        title=title,
        published=published_from_html(body, settings.date_meta_keys, settings.date_json_keys),
    )


class HttpFetcher:
    """A plain GET through an injected (rate-limited) client."""

    def __init__(self, settings: WebFetchSettings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    async def fetch(self, url: str) -> FetchedPage:
        response = await self._client.get(
            url,
            headers={"User-Agent": self._settings.user_agent},
            timeout=self._settings.timeout_seconds,
            follow_redirects=True,
        )
        return page_from(url, response.status_code, response.text, "http", self._settings)


class ReaderFetcher:
    """The same URL, read through a rendering reader service."""

    def __init__(self, settings: WebFetchSettings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    async def fetch(self, url: str) -> FetchedPage:
        response = await self._client.get(
            self._settings.reader_url_template.format(url=url),
            headers={"User-Agent": self._settings.user_agent},
            timeout=self._settings.timeout_seconds,
        )
        return page_from(url, response.status_code, response.text, "reader", self._settings)


class FallbackFetcher:
    """Plain HTTP, then the reader when the page looks script-rendered."""

    def __init__(
        self, primary: Fetcher, reader: Fetcher | None, settings: WebFetchSettings
    ) -> None:
        self._primary = primary
        self._reader = reader
        self._settings = settings

    async def fetch(self, url: str) -> FetchedPage:
        page = await self._primary.fetch(url)
        if self._reader is None or not needs_reader(page, self._settings):
            return page
        return await self._reader.fetch(url)


def needs_reader(page: FetchedPage, settings: WebFetchSettings) -> bool:
    """Whether a page's text looks like a JavaScript shell rather than content."""
    if not settings.reader_enabled or page.status >= 400:
        return False
    lowered = page.text.casefold()
    too_thin = len(page.text) < settings.min_text_chars
    return too_thin or any(marker.casefold() in lowered for marker in settings.js_markers)
