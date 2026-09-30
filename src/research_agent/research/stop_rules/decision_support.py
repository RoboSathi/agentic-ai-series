"""decision_support: stop when every option is scored on every user criterion."""

from research_agent.research.stop_rules.base import StopDecision, StopInput, all_passed

CHECK = "subject_criterion_coverage"


class EverySubjectScored:
    name = "every_subject_scored"

    def decide(self, observed: StopInput) -> StopDecision:
        passed, count = all_passed(observed.verdicts, CHECK)
        if count and passed:
            return StopDecision(stop=True, reason="every option is scored on every criterion")
        return StopDecision(stop=False, reason="some option × criterion pairs have no evidence")
