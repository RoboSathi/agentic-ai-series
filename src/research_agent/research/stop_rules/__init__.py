"""One stop-rule strategy per archetype, looked up by the Brief's archetype."""

from research_agent.domain.scope import Archetype
from research_agent.research.stop_rules.base import StopDecision, StopInput, StopRule
from research_agent.research.stop_rules.decision_support import EverySubjectScored
from research_agent.research.stop_rules.explainer import EveryConceptCovered
from research_agent.research.stop_rules.fact_verification import EveryClaimDecided
from research_agent.research.stop_rules.landscape_brief import NoNewEntities
from research_agent.research.stop_rules.question_answered import QuestionAnswered

_RULES: dict[Archetype, type[StopRule]] = {
    "decision_support": EverySubjectScored,
    "landscape_brief": NoNewEntities,
    "fact_verification": EveryClaimDecided,
    "explainer": EveryConceptCovered,
    "general": QuestionAnswered,
    "qa": QuestionAnswered,
}


def stop_rule_for(archetype: Archetype) -> StopRule:
    return _RULES[archetype]()


__all__ = [
    "EveryClaimDecided",
    "EveryConceptCovered",
    "EverySubjectScored",
    "NoNewEntities",
    "QuestionAnswered",
    "StopDecision",
    "StopInput",
    "StopRule",
    "stop_rule_for",
]
