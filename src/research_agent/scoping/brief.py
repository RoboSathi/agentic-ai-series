"""From covered checklist items to a typed Scope Brief — deterministically.

The interviewer's job ends at "this item is covered, and here is what the
user said". Turning those answers into Brief fields is parsing, not
judgement, so it happens here in code: list answers are split on a
configured pattern, a recency answer becomes a day count (or none), a
source answer becomes a policy. The same answers always make the same
Brief, which is what makes a frozen contract reproducible.
"""

import re
from collections.abc import Sequence
from typing import Any

from research_agent.config.settings import ScopingSettings
from research_agent.domain.budget import Budget
from research_agent.domain.scope import (
    Archetype,
    CoverageItem,
    KnownUnknown,
    ScopeBrief,
    SourcePolicy,
)

_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")
_AVOID = re.compile(r"\b(?:avoid|block|exclude|not|never|except)\b", re.IGNORECASE)
_CLAUSE = re.compile(r";|\.\s")


class BriefBuilder:
    """Assembles the Scope Brief from the interview's covered items."""

    def __init__(self, settings: ScopingSettings, budget: Budget) -> None:
        self._settings = settings
        self._split = re.compile(settings.answer_split_pattern, re.IGNORECASE)
        self._budget = budget

    def build(
        self,
        archetype: Archetype,
        topic: str,
        items: Sequence[CoverageItem],
        unknowns: Sequence[KnownUnknown],
    ) -> ScopeBrief:
        fields: dict[str, Any] = {}
        for item in items:
            if item.status == "covered" and item.brief_field and not self._declined(item.value):
                fields[item.brief_field] = self._value(item)
        return ScopeBrief(
            archetype=archetype,
            topic=topic,
            budget=self._budget,
            known_unknowns=tuple(unknowns),
            **fields,
        )

    def _value(self, item: CoverageItem) -> Any:
        if item.brief_field == "recency_days":
            return self._recency(item.value)
        if item.brief_field == "source_policy":
            return self._source_policy(item.value)
        if item.multi:
            return tuple(self.split(item.value))
        return item.value.strip()

    def _declined(self, answer: str) -> bool:
        """An answer that says "nothing to add" leaves the field at its default."""
        return answer.strip().casefold() in {
            word.casefold() for word in self._settings.none_answers
        }

    def split(self, answer: str) -> list[str]:
        """A list answer's parts, trimmed, empty parts dropped."""
        return [part.strip(" .") for part in self._split.split(answer) if part.strip(" .")]

    def _recency(self, answer: str) -> int | None:
        if any(word in answer.casefold() for word in self._settings.any_recency_words):
            return None
        number = _NUMBER.search(answer)
        return int(number.group()) if number else None

    def _source_policy(self, answer: str) -> SourcePolicy:
        preferred: list[str] = []
        blocked: list[str] = []
        for clause in _CLAUSE.split(answer):
            domains = [domain.casefold() for domain in _DOMAIN.findall(clause)]
            (blocked if _AVOID.search(clause) else preferred).extend(domains)
        return SourcePolicy(
            preferred_domains=tuple(preferred),
            blocked_domains=tuple(blocked),
            primary_sources_preferred=self._settings.primary_sources_phrase in answer.casefold(),
        )
