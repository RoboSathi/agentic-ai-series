"""A page's own publication date, read from its markup.

Search providers outside a news vertical (Tavily's general topic, for one) return no
date per hit, so the only trustworthy date is the one the page declares about itself:
a `<meta>` tag, a JSON-LD key, or a `<time datetime>` element. The names looked for
are configuration. The raw string is returned, unparsed; reading it is the caller's job.
"""

import re
from collections.abc import Sequence
from html.parser import HTMLParser

_TIME_TAG = "time"
_META_TAG = "meta"
_ATTRIBUTES = ("property", "name", "itemprop")


class _DateFinder(HTMLParser):
    def __init__(self, meta_keys: Sequence[str]) -> None:
        super().__init__()
        self._meta_keys = [key.casefold() for key in meta_keys]
        self.meta: dict[str, str] = {}
        self.time = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        if tag == _META_TAG:
            self._meta(values)
        elif tag == _TIME_TAG and not self.time:
            self.time = values.get("datetime", "")

    def _meta(self, values: dict[str, str]) -> None:
        for attribute in _ATTRIBUTES:
            key = values.get(attribute, "").casefold()
            if key in self._meta_keys and values.get("content"):
                self.meta.setdefault(key, values["content"])

    def first(self) -> str:
        return next((self.meta[key] for key in self._meta_keys if key in self.meta), "")


def published_from_html(html: str, meta_keys: Sequence[str], json_keys: Sequence[str]) -> str:
    """The first declared publication date in `html`, or "" when it declares none."""
    if "<" not in html:
        return ""
    finder = _DateFinder(meta_keys)
    finder.feed(html)
    return finder.first() or _json_ld(html, json_keys) or finder.time


def _json_ld(html: str, json_keys: Sequence[str]) -> str:
    if not json_keys:
        return ""
    alternatives = "|".join(re.escape(key) for key in json_keys)
    match = re.search(rf'"(?:{alternatives})"\s*:\s*"([^"]+)"', html)
    return match.group(1) if match else ""
