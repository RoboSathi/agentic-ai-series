"""Lessons across runs (Reflexion): the critics' findings, remembered as words.

Reflection inside a run fixes that run. Reflexion carries what its critics found to the next
run on a similar topic: a criterion the audit failed, the pages that could not be used and
why, a limit the run hit, a revision that raised the audit's score. Each lesson is written in
code from the run's own verdicts — never by a model, so a lesson never claims what no check
found — and appended to episodic memory when the run is archived.

At the next run's plan, lessons from earlier runs of the same kind whose topics share words
with this one are recalled and handed to the planner, the reflector and the reviser, which are
told to use them to avoid repeating a failure and never as evidence.
"""

import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field

from agent_platform.memory import EpisodicStore
from agent_platform.memory.types import MemoryFact, Provenance
from agent_platform.verification.acceptance import CheckVerdict
from research_agent.config.settings import ReflexionSettings
from research_agent.domain.rubric import Contract

_WORD = re.compile(r"[^\W\d_]+")


class RunOutcome(BaseModel):
    """What a finished run's critics found, as the lesson book reads it."""

    contract: Contract
    failed: list[CheckVerdict] = Field(default_factory=list)
    rejected_codes: dict[str, str] = Field(default_factory=dict)
    budget_stops: list[str] = Field(default_factory=list)
    audit_scores: list[float] = Field(default_factory=list)
    revisions: int = 0


class LessonBook:
    """Writes a run's lessons to episodic memory, and recalls earlier runs' for a new one."""

    def __init__(self, store: EpisodicStore, agent_id: str, settings: ReflexionSettings) -> None:
        self._store = store
        self._agent = agent_id
        self._settings = settings

    def write(self, outcome: RunOutcome) -> list[str]:
        """The run's lessons, appended to memory; returns them as written."""
        lessons = self.lessons(outcome)[: self._settings.max_lessons]
        contract = outcome.contract
        provenance = Provenance(source=contract.contract_id, kind="agent_inference")
        for index, text in enumerate(lessons, start=1):
            self._store.write(
                MemoryFact(
                    provenance=provenance,
                    key=f"{self._settings.category}:{contract.contract_id}:{index}",
                    value=text,
                    category=self._settings.category,
                    subject_id=self._agent,
                    metadata={
                        "session_id": self._settings.session,
                        "archetype": contract.brief.archetype,
                        "run": contract.contract_id,
                        "topic": contract.brief.topic,
                    },
                ),
                provenance,
            )
        return lessons

    def recall(self, contract: Contract) -> list[MemoryFact]:
        """Earlier runs' lessons for a run of this kind on a topic sharing its words."""
        terms = self._terms(contract.brief.topic)
        if not terms:
            return []
        found = self._store.search_terms(terms, self._agent, self._settings.search_limit)
        return [
            fact
            for fact in found
            if fact.category == self._settings.category
            and fact.metadata.get("archetype") == contract.brief.archetype
            and fact.metadata.get("run") != contract.contract_id
        ][: self._settings.recall_limit]

    def lessons(self, outcome: RunOutcome) -> list[str]:
        """The lessons a run's verdicts support, most useful first."""
        words, topic = self._settings.wording, outcome.contract.brief.topic
        kind = outcome.contract.brief.archetype.replace("_", " ")
        lessons = [
            words["failed"].format(kind=kind, topic=topic, criterion=_criterion(v, outcome),
                                   detail=v.detail or v.name)
            for v in outcome.failed
        ]  # fmt: skip
        codes = Counter(outcome.rejected_codes.values()).most_common(self._settings.top_codes)
        if codes:
            listed = ", ".join(f"{code} ({count})" for code, count in codes)
            lessons.append(words["rejected"].format(topic=topic, codes=listed))
        if outcome.budget_stops:
            limits = ", ".join(dict.fromkeys(outcome.budget_stops))
            lessons.append(words["budget"].format(topic=topic, limits=limits))
        if outcome.revisions and len(outcome.audit_scores) > 1:
            first, last = outcome.audit_scores[0], outcome.audit_scores[-1]
            lessons.append(
                words["revised"].format(topic=topic, n=outcome.revisions, first=first, last=last)
            )
        return lessons

    def _terms(self, topic: str) -> list[str]:
        return [
            word
            for word in dict.fromkeys(_WORD.findall(topic.casefold()))
            if len(word) >= self._settings.min_term_chars
        ]


def _criterion(verdict: CheckVerdict, outcome: RunOutcome) -> str:
    item = outcome.contract.rubric.item(verdict.item_id)
    return item.text if item is not None else verdict.name


def lesson_lines(facts: Sequence[MemoryFact]) -> list[str]:
    """Recalled lessons as the prompts carry them: the lesson, and the run it came from."""
    return [f"{fact.value} (run {fact.metadata.get('run', '')})" for fact in facts]
