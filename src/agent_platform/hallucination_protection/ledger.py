"""The fact ledger: a claim checked against typed facts before any model reads it.

The evidence that decides most of what a support reply says is structured: a row
(delivered on 2026-08-02, total $67.27), a setting (refunds settle within 3600 s), a
session (the customer is Dennis Johnson, account 2, verified). Flattened into sentences
and handed to a cross-encoder with the three that embed closest, it became a premise
lottery: "is eligible for a refund" read as contradicted by "no refund has been
requested", and "settles within 60 minutes" as unrelated to the setting that says so.

So an agent puts *facts* in its evidence — a self-contained sentence, as before, and
the typed value behind it in the chunk's metadata — and this module reads the values
back. A claim is checked by small deterministic rules, in order:

- a comparison between two values the facts hold ("$67.27 exceeds the $50.00 limit")
  is computed;
- a flag is asserted or denied ("is eligible for a refund", "is not eligible");
- a value is stated beside the words for its attribute ("delivered on 2026-08-05");
- a name or status the facts hold is stated ("You're Dennis Johnson").

Any rule that finds the claim at odds with a fact refutes it. Support needs every
specific the claim names (ids, amounts, dates, numbers) to be accounted for by a fact
that held; otherwise the ledger says nothing and the claim goes on to the state model
and NLI. Which attributes exist and the words for them are the agent's configuration
(`hallucination_protection.facts`); the platform never learns what an order is.
"""

import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.config.settings import FactAttributeSettings, FactSettings
from agent_platform.context.types import Chunk, ChunkKind
from agent_platform.hallucination_protection.claims import money_pattern, specifics
from agent_platform.primitives.money import money_or_none

# `Chunk.metadata` key holding the fact a chunk states.
FACT_KEY = "fact"
# The tier and detector a verdict names when the ledger decided it.
FACT_TIER = "facts"
FACT_MODEL = "fact-ledger"
# A claim the facts contradict.
FACT_MISMATCH = "FACT_MISMATCH"
# How a flag's value is written in a fact.
FLAG_TRUE = "true"

_OPERATORS: dict[str, Callable[[Decimal, Decimal], bool]] = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_NAMED_DATE = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\d{4}")
_NUMBER = re.compile(r"(?<![\d.,$-])#?(\d+)(?![\d]|[.,]\d)")
_COMPARABLE = ("money", "duration")


class Fact(BaseModel):
    """One typed fact: what it is about, which attribute, its value, and how it reads.

    `subject` is the record the fact is about, as the agent's identifier writes it
    (an order number); empty for a fact about the session or the rules. `value` is
    canonical for its type: money `67.27`, date `2026-08-02`, duration in seconds,
    a flag `true`/`false`, a number or text as written.
    """

    subject: str = ""
    attribute: str
    value: str
    statement: str
    kind: ChunkKind
    source: str


def fact_chunk(fact: Fact) -> Chunk:
    """The fact as evidence: its sentence for every reader, its value for this one."""
    return Chunk(
        content=fact.statement,
        kind=fact.kind,
        source=fact.source,
        metadata={FACT_KEY: fact.model_dump()},
    )


def facts_in(evidence: list[Chunk]) -> list[tuple[Fact, Chunk]]:
    """Every fact the evidence carries, with the chunk that states it."""
    return [
        (Fact.model_validate(chunk.metadata[FACT_KEY]), chunk)
        for chunk in evidence
        if isinstance(chunk.metadata.get(FACT_KEY), dict)
    ]


class LedgerFinding(NamedTuple):
    """What the facts say about a claim: whether it holds, why, and which facts said so."""

    holds: bool
    rationale: str
    chunks: list[Chunk]


class _Reading(NamedTuple):
    """One rule's reading of the claim against one or two facts."""

    holds: bool
    rationale: str
    chunks: list[Chunk]
    # The parts of the claim this reading accounted for.
    spans: list[str]


class _Value(NamedTuple):
    """A typed value written in a claim: where, as written, and its comparable form."""

    start: int
    end: int
    text: str
    number: Decimal


class FactLedger:
    """Reads a claim against the typed facts in its evidence."""

    def __init__(self, settings: FactSettings | None = None, identifier: str | None = None) -> None:
        config = get_settings().hallucination_protection
        self._settings = settings or config.facts
        pattern = identifier if identifier is not None else config.refute_identifier_pattern
        self._identifier = re.compile(pattern) if pattern else None
        self._negations = tuple(m.casefold() for m in get_settings().evals.nli.negation_markers)
        self._promise = re.compile(config.promissory_pattern, re.IGNORECASE)
        self._units = [
            (re.compile(rf"(\d+(?:\.\d+)?)\s*(?:{unit})\b", re.IGNORECASE), Decimal(seconds))
            for unit, seconds in self._settings.duration_units.items()
        ]

    def judge(self, claim: str, evidence: list[Chunk]) -> LedgerFinding | None:
        """Refuted if any fact disagrees; supported if facts account for the whole claim."""
        named = self._named(claim)
        facts = self._relevant(named, facts_in(evidence))
        if not facts:
            return None
        readings = self._readings(claim, facts)
        conflicts = [reading for reading in readings if not reading.holds]
        if conflicts:
            return LedgerFinding(False, conflicts[0].rationale, conflicts[0].chunks)
        if not readings or not self._accounted(claim, readings, named):
            return None
        chunks = list({id(c): c for r in readings for c in r.chunks}.values())
        return LedgerFinding(True, "; ".join(r.rationale for r in readings), chunks)

    def _readings(self, claim: str, facts: list[tuple[Fact, Chunk]]) -> list[_Reading]:
        compared = self._comparisons(claim, facts)
        taken = {span for reading in compared for span in reading.spans}
        return [
            *compared,
            *self._flags(claim, facts),
            *self._values(claim, facts, taken),
            *self._texts(claim, facts),
        ]

    def _named(self, claim: str) -> set[str]:
        """The record identifiers the claim names."""
        return set(self._identifier.findall(claim)) if self._identifier else set()

    @staticmethod
    def _relevant(named: set[str], facts: list[tuple[Fact, Chunk]]) -> list[tuple[Fact, Chunk]]:
        """Facts about what the claim names, or the one record in play, and every global fact.

        A claim naming no record ("it will settle within 60 minutes") is about the one
        record the facts describe; when they describe several, only global facts apply.
        """
        subjects = {fact.subject for fact, _ in facts if fact.subject}
        wanted = named or (subjects if len(subjects) == 1 else set())
        return [
            (fact, chunk) for fact, chunk in facts if not fact.subject or fact.subject in wanted
        ]

    # --- comparisons -------------------------------------------------------------

    def _comparisons(self, claim: str, facts: list[tuple[Fact, Chunk]]) -> list[_Reading]:
        """ "$67.27 exceeds the $50.00 limit": both values facts hold, the relation computed."""
        readings = []
        for symbol, phrases in self._settings.comparisons.items():
            for phrase in phrases:
                for found in re.finditer(phrase, claim, re.IGNORECASE):
                    reading = self._compare(claim, found, _OPERATORS[symbol], symbol, facts)
                    if reading is not None:
                        readings.append(reading)
        return readings

    def _compare(
        self,
        claim: str,
        found: re.Match[str],
        relation: Callable[[Decimal, Decimal], bool],
        symbol: str,
        facts: list[tuple[Fact, Chunk]],
    ) -> _Reading | None:
        for kind in _COMPARABLE:
            values = self._typed(claim, kind)
            left = [v for v in values if v.end <= found.start()]
            right = [v for v in values if v.start >= found.end()]
            if not left or not right:
                continue
            first, second = left[-1], right[0]
            held = [self._holding(kind, value, facts) for value in (first, second)]
            if held[0] is None or held[1] is None:
                return None
            holds = relation(first.number, second.number)
            verdict = "" if holds else "not "
            rationale = (
                f"{first.text} ({held[0][0].attribute}) is {verdict}{symbol} "
                f"{second.text} ({held[1][0].attribute})"
            )
            return _Reading(holds, rationale, [held[0][1], held[1][1]], [first.text, second.text])
        return None

    def _holding(
        self, kind: str, value: _Value, facts: list[tuple[Fact, Chunk]]
    ) -> tuple[Fact, Chunk] | None:
        """The fact of this type whose value is the one written."""
        return next(
            (
                (fact, chunk)
                for fact, chunk in facts
                if self._attribute(fact).type == kind and _same(kind, fact.value, value)
            ),
            None,
        )

    # --- flags -------------------------------------------------------------------

    def _flags(self, claim: str, facts: list[tuple[Fact, Chunk]]) -> list[_Reading]:
        """A flag the claim asserts or denies, against the value the fact holds."""
        readings = []
        for fact, chunk in facts:
            attribute = self._attribute(fact)
            if attribute.type != "flag":
                continue
            asserted = self._asserted(claim, attribute)
            if asserted is None:
                continue
            actual = fact.value == FLAG_TRUE
            said = "" if asserted else "not "
            rationale = f"{fact.attribute} is {fact.value}" + (
                "" if asserted == actual else f", but the reply says it is {said}so"
            )
            readings.append(_Reading(asserted == actual, rationale, [chunk], []))
        return readings

    @staticmethod
    def _asserted(claim: str, attribute: FactAttributeSettings) -> bool | None:
        """True when the claim asserts the flag, False when it denies it, None when silent."""
        if any(re.search(p, claim, re.IGNORECASE) for p in attribute.deny):
            return False
        if any(re.search(p, claim, re.IGNORECASE) for p in attribute.affirm):
            return True
        return None

    # --- typed values ------------------------------------------------------------

    def _values(
        self, claim: str, facts: list[tuple[Fact, Chunk]], taken: set[str]
    ) -> list[_Reading]:
        """A date, amount, duration or number stated beside the words for its attribute."""
        readings = []
        for fact, chunk in facts:
            attribute = self._attribute(fact)
            if attribute.type in ("flag", "text") or not self._mentions(claim, attribute):
                continue
            written = [v for v in self._typed(claim, attribute.type) if v.text not in taken]
            reading = self._value_reading(fact, chunk, attribute.type, written, facts)
            if reading is not None:
                readings.append(reading)
        return readings

    def _value_reading(
        self,
        fact: Fact,
        chunk: Chunk,
        kind: str,
        written: list[_Value],
        facts: list[tuple[Fact, Chunk]],
    ) -> _Reading | None:
        """Holds when the claim writes the fact's value; refuted when it writes an unknown one.

        A value another fact holds is not a conflict ("refund up to $50.00" names the limit,
        not a wrong total). Only dates and durations conflict here: an amount no evidence
        holds is the direct check's finding already.
        """
        matching = [v for v in written if _same(kind, fact.value, v)]
        if matching:
            spans = [v.text for v in matching]
            return _Reading(True, f"{fact.attribute} is {matching[0].text}", [chunk], spans)
        unknown = [v for v in written if self._holding(kind, v, facts) is None]
        if kind in ("date", "duration") and unknown:
            return _Reading(
                False,
                f"the record's {fact.attribute} is {fact.value}, not {unknown[0].text}",
                [chunk],
                [],
            )
        return None

    def _texts(self, claim: str, facts: list[tuple[Fact, Chunk]]) -> list[_Reading]:
        """A name or status the facts hold, stated in the claim — not denied, not promised.

        "Will be delivered" names the status the record holds and says something else.
        """
        if self._negated(claim) or self._promise.search(claim):
            return []
        return [
            _Reading(True, f"{fact.attribute} is {fact.value}", [chunk], [fact.value])
            for fact, chunk in facts
            if self._attribute(fact).type == "text"
            and re.search(rf"\b{re.escape(fact.value)}\b", claim, re.IGNORECASE)
        ]

    # --- parsing -----------------------------------------------------------------

    def _typed(self, claim: str, kind: str) -> list[_Value]:
        """Every value of this type the claim writes, in order."""
        readers: dict[str, Callable[[str], list[_Value]]] = {
            "money": _money_values,
            "date": _date_values,
            "duration": self._duration_values,
            "number": _number_values,
        }
        reader = readers.get(kind)
        return sorted(reader(claim), key=lambda v: v.start) if reader else []

    def _duration_values(self, claim: str) -> list[_Value]:
        return [
            _Value(m.start(), m.end(), m.group(0), Decimal(m.group(1)) * seconds)
            for pattern, seconds in self._units
            for m in pattern.finditer(claim)
        ]

    def _attribute(self, fact: Fact) -> FactAttributeSettings:
        """How claims state this fact's attribute; an unconfigured one is plain text."""
        return self._settings.attributes.get(fact.attribute, FactAttributeSettings(type="text"))

    @staticmethod
    def _mentions(claim: str, attribute: FactAttributeSettings) -> bool:
        return any(re.search(term, claim, re.IGNORECASE) for term in attribute.terms)

    def _negated(self, text: str) -> bool:
        lowered = text.casefold()
        return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in self._negations)

    def _accounted(self, claim: str, readings: list[_Reading], named: set[str]) -> bool:
        """Whether every specific the claim names was read from a fact that held."""
        covered = specifics(" ".join([*named, *(s for r in readings for s in r.spans)]))
        return specifics(claim) <= covered


def _same(kind: str, fact_value: str, value: _Value) -> bool:
    """Whether a written value is the fact's value, compared as its type."""
    if kind == "date":
        return bool(_days(fact_value) & _days(value.text))
    if kind == "money":
        held = money_or_none(fact_value)
        return held is not None and held == money_or_none(value.text)
    try:
        return Decimal(fact_value) == value.number
    except InvalidOperation:
        return False


def _days(text: str) -> set[str]:
    """The days a text names, in every form `specifics` reads, without bare years.

    "2026-08-02" and "2026-08-05" share the year 2026, which makes neither the other.
    """
    return {form for form in specifics(text) if not _YEAR.fullmatch(form)}


def _money_values(claim: str) -> list[_Value]:
    """Amounts, ordered by their minor units (one currency per deployment)."""
    return [
        _Value(m.start(), m.end(), m.group(0), Decimal(amount.cents))
        for m in money_pattern().finditer(claim)
        if (amount := money_or_none(m.group(0))) is not None
    ]


def _date_values(claim: str) -> list[_Value]:
    """ISO dates and month-name dates; compared by the forms `specifics` reads, not ordered."""
    found = [*_ISO_DATE.finditer(claim), *_NAMED_DATE.finditer(claim)]
    return [_Value(m.start(), m.end(), m.group(0), Decimal(0)) for m in found]


def _number_values(claim: str) -> list[_Value]:
    return [
        _Value(m.start(), m.end(), m.group(0), Decimal(m.group(1))) for m in _NUMBER.finditer(claim)
    ]


__all__ = [
    "FACT_KEY",
    "FACT_MISMATCH",
    "FACT_MODEL",
    "FACT_TIER",
    "FLAG_TRUE",
    "Fact",
    "FactLedger",
    "LedgerFinding",
    "fact_chunk",
    "facts_in",
]
