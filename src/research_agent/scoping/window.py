"""Reading a time window out of plain words: "in the last 1 month" -> 30 days.

Used twice with one rule: by the research-kind step, on the `/research` question (so "news in
the last month" is not asked about again), and by the Brief builder, on
the interview's recency answer. The units and phrasings are configured;
the order is fixed — an explicit amount beats a relative phrase, which
beats "this week", which beats "today", which beats a bare number.
"""

import calendar
import re
from datetime import date, timedelta

from research_agent.config.settings import TimeWindowSettings

_BARE_NUMBER = re.compile(r"^\s*(\d+)\s*$")


class TimeWindowParser:
    """Turns a phrase into a day count, or None when it states no window."""

    def __init__(self, settings: TimeWindowSettings) -> None:
        self._units = {unit.casefold(): days for unit, days in settings.unit_days.items()}
        self._months = {unit.casefold(): n for unit, n in settings.unit_months.items()}
        self._amount = re.compile(settings.amount_pattern, re.IGNORECASE)
        self._relative = re.compile(settings.relative_pattern, re.IGNORECASE)
        self._current = re.compile(settings.current_pattern, re.IGNORECASE)
        self._today = re.compile(settings.today_pattern, re.IGNORECASE)

    def days(self, text: str) -> int | None:
        """A window stated in words, or — for an answer to "how recent?" — a bare number."""
        stated = self.stated_days(text)
        if stated is not None:
            return stated
        bare = _BARE_NUMBER.match(text)
        return int(bare.group(1)) if bare else None

    def stated_days(self, text: str) -> int | None:
        """A window stated in words ("the last 20 years"); a bare number is not one here."""
        span = self.span(text)
        return span[0] * self._units[span[1]] if span is not None else None

    def span(self, text: str) -> tuple[int, str] | None:
        """The window as (amount, unit) — "the last 20 years" → (20, "year") — or None."""
        amount = self._amount.search(text)
        if amount is not None:
            return int(amount.group(1)), amount.group(2).casefold()
        one_unit = self._relative.search(text) or self._current.search(text)
        if one_unit is not None:
            return 1, one_unit.group(1).casefold()
        if self._today.search(text):
            return 1, self._smallest_unit()
        return None

    def start(self, today: date, span: tuple[int, str]) -> date:
        """Where the window starts, counted back on the calendar for months and years."""
        amount, unit = span
        months = self._months.get(unit)
        if months is None:
            return today - timedelta(days=amount * self._units[unit])
        total = today.year * 12 + today.month - 1 - amount * months
        year, month = divmod(total, 12)
        day = min(today.day, calendar.monthrange(year, month + 1)[1])
        return date(year, month + 1, day)

    def _smallest_unit(self) -> str:
        return min(self._units, key=lambda unit: self._units[unit])
