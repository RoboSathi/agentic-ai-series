"""Reading a time window out of plain words: "in the last 1 month" -> 30 days.

Used twice with one rule: by the gate, on the first message (so "news in
the last month" is not asked about again), and by the Brief builder, on
the interview's recency answer. The units and phrasings are configured;
the order is fixed — an explicit amount beats a relative phrase, which
beats "this week", which beats "today", which beats a bare number.
"""

import re

from research_agent.config.settings import TimeWindowSettings

_BARE_NUMBER = re.compile(r"^\s*(\d+)\s*$")


class TimeWindowParser:
    """Turns a phrase into a day count, or None when it states no window."""

    def __init__(self, settings: TimeWindowSettings) -> None:
        self._units = {unit.casefold(): days for unit, days in settings.unit_days.items()}
        self._amount = re.compile(settings.amount_pattern, re.IGNORECASE)
        self._relative = re.compile(settings.relative_pattern, re.IGNORECASE)
        self._current = re.compile(settings.current_pattern, re.IGNORECASE)
        self._today = re.compile(settings.today_pattern, re.IGNORECASE)

    def days(self, text: str) -> int | None:
        amount = self._amount.search(text)
        if amount is not None:
            return int(amount.group(1)) * self._unit(amount.group(2))
        one_unit = self._relative.search(text) or self._current.search(text)
        if one_unit is not None:
            return self._unit(one_unit.group(1))
        if self._today.search(text):
            return 1
        bare = _BARE_NUMBER.match(text)
        return int(bare.group(1)) if bare else None

    def _unit(self, name: str) -> int:
        return self._units[name.casefold()]
