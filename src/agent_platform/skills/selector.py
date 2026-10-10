"""Choosing one turn's procedures, deterministically (ADR 0010).

No model call decides which procedure a prompt carries. A skill declared
`always` joins every turn; a skill whose trigger matches one of the
turn's signals joins it; and only when no trigger matched does the query
itself select, by BM25 over each skill's description and keywords. A
signal is an opaque string the agent derives (`intent:REQUEST_REFUND`):
it is matched here, never interpreted.
"""

from collections.abc import Sequence

from agent_platform.config.settings import SkillSettings
from agent_platform.context.retrieval import LexicalRetriever
from agent_platform.context.types import Chunk
from agent_platform.skills.types import ActivatedSkill, Skill

# A signal is `kind:value`; its reason reads "matched intent REQUEST_REFUND".
_SIGNAL_SEPARATOR = ":"


class SkillSelector:
    """always ∪ triggered, else lexical — capped."""

    def __init__(self, settings: SkillSettings) -> None:
        self._always = settings.always_trigger
        self._reasons = settings.reasons
        self._selection = settings.selection

    def select(
        self, skills: Sequence[Skill], signals: frozenset[str], query: str
    ) -> tuple[ActivatedSkill, ...]:
        """The procedures this turn's prompt carries, each with its reason."""
        declared = [chosen for skill in skills if (chosen := self._declared(skill, signals))]
        if not any(chosen.reason != self._reasons.always for chosen in declared):
            taken = {chosen.skill.name for chosen in declared}
            declared += self._lexical([s for s in skills if s.name not in taken], query)
        return tuple(declared[: self._selection.max_active])

    def _declared(self, skill: Skill, signals: frozenset[str]) -> ActivatedSkill | None:
        if self._always in skill.document.triggers:
            return ActivatedSkill(skill=skill, reason=self._reasons.always)
        for trigger in skill.document.triggers:
            if trigger in signals:
                kind, _, value = trigger.partition(_SIGNAL_SEPARATOR)
                reason = self._reasons.trigger.format(kind=kind, value=value, signal=trigger)
                return ActivatedSkill(skill=skill, reason=reason)
        return None

    def _lexical(self, skills: list[Skill], query: str) -> list[ActivatedSkill]:
        """Skills the query names, best first, above the configured floor."""
        if not skills or not query.strip():
            return []
        retriever = LexicalRetriever([Chunk(content=_searchable(skill)) for skill in skills])
        scored = [(retriever.score(query, index), skill) for index, skill in enumerate(skills)]
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        return [
            ActivatedSkill(skill=skill, reason=self._reasons.lexical.format(score=score))
            for score, skill in ranked
            if score >= self._selection.min_lexical_score
        ]


def _searchable(skill: Skill) -> str:
    """What a query is matched against: the purpose, never the steps."""
    return " ".join([skill.document.description, *skill.document.keywords])
