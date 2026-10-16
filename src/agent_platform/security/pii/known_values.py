"""Personal data the harness already knows, found by exact match rather than guessed by a model.

Most of what an agent holds about a person it read from its own records: the verified
customer's name and email came from the database, not from anything typed. A statistical
recognizer guesses from context and misses a lone, unusual name in a fragment —
"Pythagoras (account #19)" carries no context for it — while the harness knew the value the
moment it read the record. So an agent registers each value it fetched, and every later pass
finds that value wherever it appears, however unusual, with certainty.

The registry is held in this process's memory only and never written: it would be a list of
the very values it protects. A value detected by a model at the source is registered too, so
a later copy of it (a log line written before its text was tokenised) is found without the
model.
"""

import re
import threading
from collections import OrderedDict
from functools import cache
from typing import Any

from agent_platform.config import get_settings
from agent_platform.security.pii.types import PIISpan

# How a span found here names its recognizer on a PII card.
KNOWN_VALUE_DETECTOR = "KnownValue"


class KnownValues:
    """Values to find wherever they appear, each with its category (a Presidio entity name)."""

    def __init__(self, min_chars: int, max_entries: int) -> None:
        self._min_chars = min_chars
        self._max_entries = max_entries
        self._values: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._pattern: re.Pattern[str] | None = None
        # Moves on every new value, so a cache of what a text tokenised to can tell it is stale.
        self.version = 0

    def register(self, category: str, value: str) -> None:
        """Remember `value` as `category`; one too short to be told from ordinary words is not."""
        text = value.strip()
        if len(text) < self._min_chars:
            return
        with self._lock:
            key = text.casefold()
            if self._values.get(key) == category:
                self._values.move_to_end(key)
                return
            self._values[key] = category
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)
            self._pattern = None
            self.version += 1

    def find(self, text: str) -> list[PIISpan]:
        """Every registered value in `text`, word-bounded and ignoring case."""
        pattern = self._compiled()
        if pattern is None:
            return []
        return [
            PIISpan(
                category=self._values.get(match.group().casefold(), ""),
                start=match.start(),
                end=match.end(),
                text=match.group(),
                detector=KNOWN_VALUE_DETECTOR,
            )
            for match in pattern.finditer(text)
        ]

    def _compiled(self) -> re.Pattern[str] | None:
        with self._lock:
            if self._pattern is None and self._values:
                # Longest first, so a full name wins over the first name it starts with.
                alternatives = sorted(self._values, key=len, reverse=True)
                body = "|".join(re.escape(value) for value in alternatives)
                self._pattern = re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)
            return self._pattern


class KnownValueDetector:
    """A `PIIDetector` over the process's known values."""

    def __init__(self, values: KnownValues | None = None) -> None:
        self._values = values or known_values()

    def detect(self, text: str) -> list[PIISpan]:
        return self._values.find(text) if text else []


@cache
def known_values() -> KnownValues:
    """The process's one registry: an agent registers into it, every detector reads it."""
    settings = get_settings().security.pii.known_values
    return KnownValues(settings.min_chars, settings.max_entries)


def register_known(category: str, value: str) -> None:
    """Remember a value this harness read from a record, so every later pass finds it."""
    known_values().register(category, value)


def register_rows(query: str, rows: list[dict[str, Any]]) -> None:
    """Register each personal value a query read, from the columns an agent declared.

    `security.pii.known_columns` names, per table, which columns hold what (a customer table's
    name, email and phone). A query is matched to a table by the name it reads from.
    """
    for table, columns in get_settings().security.pii.known_columns.items():
        if not _reads(query, table):
            continue
        for row in rows:
            for column, category in columns.items():
                value = row.get(column)
                if value:
                    register_known(category, str(value))


def _reads(query: str, table: str) -> bool:
    return re.search(rf'\bFROM\s+"?{re.escape(table)}"?\b', query, re.IGNORECASE) is not None


__all__ = [
    "KNOWN_VALUE_DETECTOR",
    "KnownValueDetector",
    "KnownValues",
    "known_values",
    "register_known",
    "register_rows",
]
