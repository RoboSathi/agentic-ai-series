"""An undated question asks for the latest: the check that holds a search to that.

A model asked "India's medal tally at the Asian Games" in 2026 searched for "... 2023 Hangzhou" —
the edition it knew from training — although its prompt said what today was. The prompt rule
("no date named: the latest") is necessary and not enough, so this reads what was actually
searched and what came back:

    pinned   the query names a past year the question never did
    stale    every dated result is older than `stale_after_days`

Either way the search still runs; the verdict is a note the model reads beside the results (the
direct answer), or a line on the SEARCH card and in the log (a research run, whose queries the
planner wrote from a contract the user approved — they are shown, never rewritten). A question
that names any date or window is the user's own choice, and is left alone.
"""

import re
from collections.abc import Sequence
from datetime import date
from typing import NamedTuple

from agent_platform.logging import get_logger, instrument
from research_agent.config.settings import RecencySettings
from research_agent.scoping.window import TimeWindowParser

# How many leading characters of a published date are its ISO day (YYYY-MM-DD).
_ISO_DAY = len("YYYY-MM-DD")


class RecencyVerdict(NamedTuple):
    """What one search made of an undated question: past years it pinned, its newest result."""

    pinned: tuple[int, ...] = ()
    newest: date | None = None
    stale: bool = False

    @property
    def flagged(self) -> bool:
        return bool(self.pinned) or self.stale


@instrument("research_agent")
class RecencyGuard:
    """Reads a query and its results against the question that asked for them."""

    def __init__(self, settings: RecencySettings, window: TimeWindowParser) -> None:
        self._settings = settings
        self._window = window
        self._year = re.compile(settings.year_pattern)
        self._logger = get_logger("research_agent")

    def dated(self, question: str) -> bool:
        """Whether the question names a year or a time window of its own."""
        return bool(self._year.search(question)) or self._window.days(question) is not None

    def check(
        self, question: str, query: str, published: Sequence[str], today: date
    ) -> RecencyVerdict:
        """The verdict for one search; an empty one for a dated question or a disabled check."""
        if not self._settings.enabled or self.dated(question):
            return RecencyVerdict()
        pinned = tuple(
            sorted({int(year) for year in self._year.findall(query) if int(year) < today.year})
        )
        newest = max(filter(None, (_day(value) for value in published)), default=None)
        stale = newest is not None and (today - newest).days > self._settings.stale_after_days
        verdict = RecencyVerdict(pinned=pinned, newest=newest, stale=stale)
        if verdict.flagged:
            self._logger.info(
                "Search not held to the latest",
                question=question, query=query, pinned=list(pinned),
                newest=str(newest or ""), today=str(today), stale=stale,
            )  # fmt: skip
        return verdict

    def outside_period(self, query: str, start: date) -> tuple[int, ...]:
        """The years a query names that fall before the brief's period (none: within it)."""
        if not self._settings.enabled:
            return ()
        early = tuple(sorted({int(y) for y in self._year.findall(query) if int(y) < start.year}))
        if early:
            self._logger.info(
                "Search reaches before the period", query=query, years=list(early),
                period_start=str(start),
            )  # fmt: skip
        return early

    def period_note(self, years: tuple[int, ...], start: date, end: date) -> str:
        """What the card says of a query that reaches before the period ("" when it does not)."""
        if not years:
            return ""
        return self._settings.messages.outside_period.format(
            years=", ".join(str(year) for year in years), start=start, end=end
        )

    def note(self, verdict: RecencyVerdict, today: date, today_text: str) -> str:
        """What the model reads beside the results ("" when the search was held to the latest)."""
        words = self._settings.messages
        if verdict.pinned:
            years = ", ".join(str(year) for year in verdict.pinned)
            return words.pinned.format(years=years, today=today_text, current=today.year)
        if verdict.stale and verdict.newest is not None:
            return words.stale.format(newest=verdict.newest.isoformat(), today=today_text)
        return ""


def _day(value: str) -> date | None:
    """The calendar day an ISO date or timestamp names; None when it names none."""
    try:
        return date.fromisoformat(value[:_ISO_DAY])
    except ValueError:
        return None


__all__ = ["RecencyGuard", "RecencyVerdict"]
