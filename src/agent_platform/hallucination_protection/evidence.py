"""Tool results as grounding evidence: one fact per premise, in the order they happened.

The checker compares a claim with one premise at a time, so a tool's output is
not one chunk. A multi-line result (a list of orders, a forecast by day) becomes
one chunk per line, and a line becomes one chunk per sentence: "Order 10693 has
been cancelled. Order 10693 is awaiting review." read as *neutral* on "your order
has been cancelled" when it was one premise. A refused call stays whole, so its
NOT EXECUTED marker is never parted from what was refused.

Every agent that grounds a reply on its tools builds its evidence here.
"""

from collections.abc import Sequence
from typing import NamedTuple

from agent_platform.context.types import Chunk, ChunkKind
from agent_platform.hallucination_protection.claims import split_sentences
from agent_platform.hallucination_protection.direct import NOT_EXECUTED_MARKER
from agent_platform.hallucination_protection.resolver import (
    OBSERVATION_SEQUENCE,
    STATE_CHANGING,
)

TOOL_RESULT_KIND: ChunkKind = "tool_result"


class Observation(NamedTuple):
    """What one tool call returned, whether it changed anything, and where it came from."""

    text: str
    changed: bool = False
    source: str = ""


def tool_evidence(observations: Sequence[Observation]) -> list[Chunk]:
    """The facts the tools returned, each its own `tool_result` premise, in call order."""
    return [
        Chunk(
            content=fact,
            kind=TOOL_RESULT_KIND,
            source=observation.source,
            metadata={OBSERVATION_SEQUENCE: index, STATE_CHANGING: observation.changed},
        )
        for index, observation in enumerate(observations)
        for line in observation.text.splitlines()
        if line.strip()
        for fact in observation_facts(line.strip())
    ]


def observation_facts(line: str) -> list[str]:
    """One observation line as the single facts NLI can read, one premise each."""
    if line.startswith(NOT_EXECUTED_MARKER):
        return [line]
    return [sentence for sentence in split_sentences(line) if sentence.strip()]


__all__ = ["TOOL_RESULT_KIND", "Observation", "observation_facts", "tool_evidence"]
