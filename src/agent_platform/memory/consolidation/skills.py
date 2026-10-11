"""Skill candidates from refusals that keep recurring.

Generative Agents reflect once enough has accumulated; this is the same
idea without a model. When the same tool is refused for the same reason
in enough separate sessions, that is a lesson about how to handle such
requests — and a candidate skill. It is only ever a candidate: the
dream proposes it as an agent, the agent holds no grant to create
skills, and governance queues it for a curator.
"""

import re

from pydantic import BaseModel

from agent_platform.memory.consolidation.summaries import tool_name
from agent_platform.memory.deterministic import normalize_key
from agent_platform.memory.events import MemoryEvent

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
# Numbers — order ids, amounts, dates — differ between two instances of the
# same refusal. Masked, "Order 10515 is already shipped" and "Order 10593
# is already shipped" are one recurring pattern rather than two one-offs.
_SPECIFIC = re.compile(r"\$?\d[\d,.:/-]*")
_MASK = "N"


class SkillCandidate(BaseModel):
    """A recurring refusal, worded as the skill it suggests."""

    key: str
    value: str
    tool: str
    reason: str
    sessions: int


def recurring_refusals(
    sessions: dict[str, list[MemoryEvent]], markers: list[str], threshold: int, key_chars: int
) -> list[SkillCandidate]:
    """Refusals of one tool for one reason seen in at least `threshold` sessions."""
    seen: dict[tuple[str, str], set[str]] = {}
    for session, events in sessions.items():
        for event in events:
            for action in event.actions:
                reason = _SPECIFIC.sub(_MASK, _refusal_reason(str(action.result or ""), markers))
                if reason:
                    tool = tool_name(action.name)
                    seen.setdefault((tool, reason), set()).add(session)
    return [
        SkillCandidate(
            key=normalize_key(f"{tool} refused {reason}")[:key_chars],
            value=(
                f"'{tool}' was refused in {len(where)} sessions because: {reason} "
                "Check this condition and tell the customer before proposing the action."
            ),
            tool=tool,
            reason=reason,
            sessions=len(where),
        )
        for (tool, reason), where in sorted(seen.items())
        if len(where) >= threshold
    ]


def _refusal_reason(result: str, markers: list[str]) -> str:
    """The first sentence after a refusal marker, or "" if this is not a refusal."""
    for marker in markers:
        if result.startswith(marker):
            rest = result[len(marker) :].strip()
            return _SENTENCE_END.split(rest, maxsplit=1)[0]
    return ""
