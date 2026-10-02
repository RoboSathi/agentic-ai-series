"""From covered checklist items to a typed Scope Brief — deterministically.

The interviewer's job ends at "this item is covered, and here is what the
user said". Turning those answers into Brief fields is parsing, not
judgement, so it happens here in code: list answers are split on a
configured pattern, a recency answer becomes a day count (or none), a
source answer becomes a policy. The same answers always make the same
Brief, which is what makes a frozen contract reproducible.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from agentic_commons.logging import instrument
from research_agent.config.settings import DepthProfile, ScopingSettings
from research_agent.domain.budget import Budget
from research_agent.domain.scope import (
    DEFAULT_DEPTH,
    QA,
    Archetype,
    CoverageItem,
    Depth,
    KnownUnknown,
    ScopeBrief,
    SourcePolicy,
)
from research_agent.scoping.window import TimeWindowParser

_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)
_AVOID = re.compile(r"\b(?:avoid|block|exclude|not|never|except)\b", re.IGNORECASE)
_CLAUSE = re.compile(r";|\.\s")
DEPTH_FIELD = "depth"


@instrument("research_agent")
class BriefBuilder:
    """Assembles the Scope Brief from the interview's covered items."""

    def __init__(
        self,
        settings: ScopingSettings,
        budget: Budget,
        qa_budget: Budget,
        depths: Mapping[Depth, DepthProfile],
    ) -> None:
        self._settings = settings
        self._split = re.compile(settings.answer_split_pattern, re.IGNORECASE)
        self._pages = re.compile(settings.pages_pattern, re.IGNORECASE)
        self._budget = budget
        self._qa_budget = qa_budget
        self._depths = depths
        self._window = TimeWindowParser(settings.time_window)

    def question(self, topic: str, recent_days: int | None) -> ScopeBrief:
        """The Q&A Brief: the question and its time window, nothing to interview for."""
        return ScopeBrief(
            archetype=QA,
            topic=topic,
            recency_days=recent_days,
            budget=self._qa_budget.capped_by(self._budget),
        )

    def build(
        self,
        archetype: Archetype,
        topic: str,
        items: Sequence[CoverageItem],
        unknowns: Sequence[KnownUnknown],
    ) -> ScopeBrief:
        fields: dict[str, Any] = {}
        size = ""
        for item in items:
            if item.status != "covered" or not item.brief_field or self._declined(item.value):
                continue
            if item.brief_field == DEPTH_FIELD:
                size = item.value
            else:
                fields[item.brief_field] = self._value(item)
        depth, budget = self._sized(size)
        return ScopeBrief(
            archetype=archetype,
            topic=topic,
            budget=budget,
            depth=depth,
            known_unknowns=tuple(unknowns),
            **fields,
        )

    def _sized(self, answer: str) -> tuple[Depth, Budget]:
        """The size the user asked for (small when they did not) and the budget it allows.

        "N pages" sets the fetch cap itself, on the smallest size that can read that many.
        Whatever is asked, the agent's own budget is still the ceiling.
        """
        wanted = self._pages.search(answer)
        depth = self._depth_named(answer) or DEFAULT_DEPTH
        if wanted:
            pages = int(wanted.group(1))
            depth = self._smallest_reading(pages)
            budget = self._depths[depth].budget.model_copy(update={"max_fetches": pages})
        else:
            budget = self._depths[depth].budget
        return depth, budget.capped_by(self._budget)

    def _depth_named(self, answer: str) -> Depth | None:
        words = set(re.findall(r"[a-z]+", answer.casefold()))
        for depth, names in self._settings.depth_words.items():
            if words & {name.casefold() for name in names}:
                return depth
        return None

    def _smallest_reading(self, pages: int) -> Depth:
        by_fetches = sorted(self._depths, key=lambda d: self._depths[d].budget.max_fetches)
        return next(
            (d for d in by_fetches if self._depths[d].budget.max_fetches >= pages), by_fetches[-1]
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
        return self._window.days(answer)

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
