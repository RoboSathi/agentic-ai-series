"""landscape_brief: stop when the last batch added no new entities."""

from research_agent.research.stop_rules.base import StopDecision, StopInput


class NoNewEntities:
    name = "no_new_entities"

    def decide(self, observed: StopInput) -> StopDecision:
        if observed.entities_after > 0 and observed.entities_after == observed.entities_before:
            return StopDecision(stop=True, reason="the last batch found no new players")
        added = observed.entities_after - observed.entities_before
        return StopDecision(stop=False, reason=f"the last batch found {added} new player(s)")
