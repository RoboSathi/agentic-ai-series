"""Every figure in an answer must be one a query returned (the `hallucination_guard` feature).

A question about the agents is a question about numbers, and the failure that
matters is a number the model made up — a p95 it never computed, a count it
rounded from nothing. So the check is deterministic and narrow: each figure in
the answer is looked for among the values the queries returned (and among the
figures inside their text, so a model name's digits are not flagged), within a
relative tolerance for rounding, and as a percentage of a ratio. Dates and times
are not figures. What is left over is reported, not silently removed.
"""

import json
import re
from collections.abc import Iterable

from pydantic import BaseModel

from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord

# Not figures: dates, times of day, and ordinals ("the 95th percentile").
_NOT_FIGURES = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"  # a date, maybe with its time
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"  # a time of day
    r"|\b\d+(?:st|nd|rd|th)\b"  # an ordinal
)
# A figure: thousands grouped by commas or spaces ("121,322", "121 322"), or plain digits.
# A percentage may be written with a space before its sign ("70.6 %").
_FIGURE = re.compile(r"(?<![\w.])(?:\d{1,3}(?:[, ]\d{3})+|\d+)(?:\.\d+)?(?:\s?%)?")
_GROUPING = re.compile(r"[, ]")
_PERCENT = 100.0


class FigureFinding(BaseModel):
    """One figure of an answer, and the query whose result holds it (None: no query does)."""

    figure: str
    query: int | None = None

    @property
    def supported(self) -> bool:
        return self.query is not None


class FigureCheck:
    """Finds the figures in an answer that no query result supports."""

    def __init__(self, settings: OpsAgentSettings) -> None:
        self._tolerance = settings.number_tolerance
        self._free = set(settings.free_numbers)

    def unsupported(self, answer: str, queries: Iterable[QueryRecord]) -> list[str]:
        """The answer's figures with no matching value in any query result."""
        return [f.figure for f in self.findings(answer, queries) if not f.supported]

    def findings(self, answer: str, queries: Iterable[QueryRecord]) -> list[FigureFinding]:
        """Every figure the answer states, each with the (1-based) query that supports it."""
        evidence = [_evidence(query) for query in queries]
        return [
            FigureFinding(figure=token, query=self._source(token, value, evidence))
            for token, value in _figures(answer)
            if value not in self._free
        ]

    def _source(self, token: str, value: float, evidence: list[list[float]]) -> int | None:
        found = (n for n, values in enumerate(evidence, 1) if self._supported(token, value, values))
        return next(found, None)

    def _supported(self, token: str, value: float, evidence: list[float]) -> bool:
        candidates = [value, value / _PERCENT] if token.endswith("%") else [value]
        return any(self._close(candidate, known) for candidate in candidates for known in evidence)

    def _close(self, figure: float, known: float) -> bool:
        return abs(figure - known) <= self._tolerance * max(1.0, abs(known))


def _figures(text: str) -> list[tuple[str, float]]:
    plain = _NOT_FIGURES.sub(" ", text)
    return [
        (match.group(0), float(_GROUPING.sub("", match.group(0).rstrip("%").rstrip())))
        for match in _FIGURE.finditer(plain)
    ]


def _evidence(query: QueryRecord) -> list[float]:
    """Every value one query returned, and the figures inside its text cells."""
    values = [float(cell) for row in query.rows for cell in row if _is_number(cell)]
    values.extend(value for _, value in _figures(json.dumps(query.rows, default=str)))
    return values


def _is_number(cell: object) -> bool:
    return isinstance(cell, int | float) and not isinstance(cell, bool)


__all__ = ["FigureCheck", "FigureFinding"]
