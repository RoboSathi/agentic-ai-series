"""This agent's Tier-2 facts in the generated analytics world: reviews of queued refunds."""

import math
import random
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

from agent_platform.observability.facts import Fact
from customer_support_agent.analytics.facts import ReviewFact

# Spread of a simulated wait around its mean (log-normal sigma).
_WAIT_SIGMA = 0.8
_SECONDS_PER_MINUTE = 60


def reviews(
    tasks: Sequence[Mapping[str, Any]], params: Mapping[str, Any], rng: random.Random
) -> list[Fact]:
    """One reviewer decision per refund queued for review, some time after it was queued."""
    decisions: Mapping[str, float] = params["decisions"]
    mean_wait = float(params["wait_minutes"]) * _SECONDS_PER_MINUTE
    facts: list[Fact] = []
    for index, task in enumerate(tasks):
        decision = rng.choices(list(decisions), weights=list(decisions.values()))[0]
        wait = rng.lognormvariate(math.log(mean_wait), _WAIT_SIGMA)
        facts.append(
            ReviewFact(
                agent_id=task["agent_id"],
                session_id=task["session_id"],
                task_id=task["task_id"],
                channel=task["channel"],
                recorded_at=task["recorded_at"] + timedelta(seconds=wait),
                request_id=str(index + 1),
                decision=decision,
                refund_issued=decision == params["issued_decision"],
                wait_seconds=wait,
            )
        )
    return facts


__all__ = ["reviews"]
