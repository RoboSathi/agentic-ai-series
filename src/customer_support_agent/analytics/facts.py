"""The human review of a queued refund, as a fact (Tier 2: only this agent has reviews)."""

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import yaml

from agent_platform.observability.facts import Fact, fact_store
from agent_platform.observability.types import utc_now

# This agent's tables carry its id (ADR 0009 D7), so no two agents' tables can collide.
# The id is read from this agent's own config, not the process's settings: the table is
# the same whichever agent's overlay a process (an exporter, a generator) has loaded.
_TABLE = "{agent}__fact_review"
_OWN_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
AGENT_ID: str = yaml.safe_load(_OWN_CONFIG.read_text(encoding="utf-8"))["agent"]["id"]


class ReviewFact(Fact):
    """One reviewer decision on a refund this agent queued."""

    table: ClassVar[str] = _TABLE.format(agent=AGENT_ID)

    request_id: str = ""
    # approved, denied, failed (approved but not completed) or pending.
    decision: str = ""
    refund_issued: bool | None = None
    # From queueing to the decision; NULL when the queue time was not recorded.
    wait_seconds: float | None = None


def record_review(pending: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Append the review of `pending` (what `act` queued) with the reviewer's `verdict`."""
    issued = verdict.get("refund_issued")
    fact_store().append(
        ReviewFact(
            request_id=str(pending.get("request_id", "")),
            decision=str(verdict.get("decision", "")),
            refund_issued=bool(issued) if issued is not None else None,
            wait_seconds=_waited(pending.get("queued_at")),
        )
    )


def _waited(queued_at: object) -> float | None:
    if not isinstance(queued_at, str) or not queued_at:
        return None
    return (utc_now() - datetime.fromisoformat(queued_at)).total_seconds()


__all__ = ["ReviewFact", "record_review"]
