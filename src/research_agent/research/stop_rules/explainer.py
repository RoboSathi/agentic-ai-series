"""explainer: stop when every sub-concept is covered and cross-checked."""

from research_agent.research.stop_rules.base import StopDecision, StopInput, all_passed

CHECK = "concept_covered"


class EveryConceptCovered:
    name = "every_concept_covered"

    def decide(self, observed: StopInput) -> StopDecision:
        passed, count = all_passed(observed.verdicts, CHECK)
        if count and passed:
            return StopDecision(stop=True, reason="every sub-concept is covered in two sources")
        return StopDecision(stop=False, reason="some sub-concepts are not yet cross-checked")
