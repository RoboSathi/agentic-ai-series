"""general and qa: stop once every coverage check passes (at least one cited finding).

Neither kind has subjects to tick off one by one, so "done" is the rubric's
own coverage checks: findings exist, from enough domains, on every stated
criterion. The round cap stops a question that cannot be answered.
"""

from research_agent.research.stop_rules.base import StopDecision, StopInput


class QuestionAnswered:
    name = "question_answered"

    def decide(self, observed: StopInput) -> StopDecision:
        unmet = [v.item_id for v in observed.verdicts if not v.passed]
        if observed.verdicts and not unmet:
            return StopDecision(stop=True, reason="the question is answered with cited findings")
        return StopDecision(
            stop=False, reason=f"not yet covered: {', '.join(unmet) or 'nothing found'}"
        )
