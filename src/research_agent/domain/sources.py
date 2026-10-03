"""Facts about a source that need no run to decide: its host, its date, whether it is primary.

Pure functions over a URL or a date string, shared by the research engine,
the synthesis and the audit. They live here, below all three, so none of
those layers has to import another to ask what a URL's host is.
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlparse


def host_of(url: str) -> str:
    host = (urlparse(url).hostname or url).casefold()
    return host.removeprefix("www.")


WindowVerdict = Literal["in", "out", "undated"]


def parse_published(source_date: str) -> date | None:
    """A provider's publication date as a calendar date, or None when it cannot be read.

    Providers disagree on format: DuckDuckGo news returns ISO 8601, Tavily's news
    vertical returns RFC 2822 (`Tue, 29 Sep 2026 13:53:01 GMT`). Both are read here;
    anything else is "unknown", never silently "old".
    """
    text = (source_date or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError):
        return None


def window_verdict(source_date: str, as_of: date, days: int) -> WindowVerdict:
    """Whether a source's date is inside the window, outside it, or unreadable."""
    published = parse_published(source_date)
    if published is None:
        return "undated"
    return "in" if published >= as_of - timedelta(days=days) else "out"


def within_window(source_date: str, as_of: date, days: int) -> bool:
    """Whether a source's date falls inside a recency window; an undated source does not."""
    return window_verdict(source_date, as_of, days) == "in"


def is_primary(url: str, markers: Sequence[str]) -> bool:
    """Whether a source counts as primary under the configured host markers."""
    host = (urlparse(url).hostname or "").casefold()
    return any(marker.casefold() in host for marker in markers)
