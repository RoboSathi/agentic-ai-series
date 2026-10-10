"""The hallucination check in levels: the same first level for every agent, then the agent's own.

Level 1 is the platform's: how sure the model was of the tokens it wrote
(`confidence.py`). Level 2 is whatever grounds this agent's claims — NLI
entailment against the evidence for a reply, a figure found in a query result
for an answer computed from one. Each level is reported as a `CheckLevel`,
and the levels travel, level 1 first, inside the one `hallucination_check` event
an agent emits, so every agent's card and log line read the same way.
"""

from pydantic import BaseModel, Field

from agent_platform.observability import checks

# The one stage every agent's hallucination check is reported under.
HALLUCINATION_STAGE = "hallucination_check"
CONFIDENCE_LEVEL = 1
GROUNDING_LEVEL = 2


class CheckLevel(BaseModel):
    """One level of the hallucination check: what it is, its verdict, and its proof lines."""

    level: int
    name: str
    verdict: str
    summary: str
    lines: list[str] = Field(default_factory=list)


def grounding_level(
    name: str, summary: str, *, flagged: bool, lines: list[str] | None = None
) -> CheckLevel:
    """Level 2, as the agent's own check describes it."""
    return CheckLevel(
        level=GROUNDING_LEVEL,
        name=name,
        verdict=checks.FLAG if flagged else checks.PASS,
        summary=summary,
        lines=lines or [],
    )


__all__ = [
    "CONFIDENCE_LEVEL",
    "CheckLevel",
    "GROUNDING_LEVEL",
    "HALLUCINATION_STAGE",
    "grounding_level",
]
