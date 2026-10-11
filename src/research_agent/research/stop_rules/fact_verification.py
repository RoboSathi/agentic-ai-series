"""fact_verification: stop when every claim is supported, refuted, or can only be marked unresolved.

"Unresolved" is reached when the reflector has nothing left to try for a
claim; this rule stops early only when every claim already has its
independent sources.
"""

from research_agent.research.stop_rules.base import StopDecision, StopInput, all_passed

CHECK = "claim_verdict"


class EveryClaimDecided:
    name = "every_claim_decided"

    def decide(self, observed: StopInput) -> StopDecision:
        passed, count = all_passed(observed.verdicts, CHECK)
        if count and passed:
            return StopDecision(
                stop=True, reason="every claim is supported by two independent domains that agree"
            )
        return StopDecision(stop=False, reason="some claims lack two agreeing independent domains")
