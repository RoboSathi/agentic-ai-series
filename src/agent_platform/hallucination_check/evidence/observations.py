"""Tool results as grounding evidence: one fact per premise, in the order they happened.

The checker compares a claim with one premise at a time, so a tool's output is
not one chunk. A multi-line result (a list of orders, a forecast by day) becomes
one chunk per line, and a line becomes one chunk per sentence: "Order 10693 has
been cancelled. Order 10693 is awaiting review." read as *neutral* on "your order
has been cancelled" when it was one premise. A refused call stays whole, so its
NOT EXECUTED marker is never parted from what was refused.

Every agent that grounds a reply on its tools builds its evidence here.
"""

import re
from collections.abc import Sequence
from typing import NamedTuple

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk, ChunkKind
from agent_platform.hallucination_check.claims import plain_text, split_sentences
from agent_platform.hallucination_check.direct import NOT_EXECUTED_MARKER
from agent_platform.hallucination_check.resolver import (
    OBSERVATION_SEQUENCE,
    PASSAGE,
    STATE_CHANGING,
)

TOOL_RESULT_KIND: ChunkKind = "tool_result"


class Observation(NamedTuple):
    """What one tool call returned, whether it changed anything, and where it came from."""

    text: str
    changed: bool = False
    source: str = ""


def tool_evidence(observations: Sequence[Observation], window: int | None = None) -> list[Chunk]:
    """The facts the tools returned, each its own `tool_result` premise, in call order.

    With a `window` above 1 (`hallucination_check.passage_window`), every run of that many
    consecutive facts of one result is a premise too, marked `PASSAGE`: a page states a person,
    their role and their past roles on separate lines, and a sentence that joins them is entailed
    by none of the lines alone. A write's result is never windowed: its facts are the record.
    """
    settings = get_settings().hallucination_check
    size = window or settings.passage_window
    pages = settings.tool_results_are_pages
    chunks: list[Chunk] = []
    for index, observation in enumerate(observations):
        lines = _page_lines(observation.text) if pages else observation.text.splitlines()
        facts = [fact for line in lines if line.strip() for fact in observation_facts(line.strip())]
        if pages:
            facts = list(dict.fromkeys(facts))
        metadata: dict[str, object] = {
            OBSERVATION_SEQUENCE: index,
            STATE_CHANGING: observation.changed,
        }
        chunks.extend(_chunk(fact, observation, metadata) for fact in facts)
        if size > 1 and not observation.changed:
            chunks.extend(
                _chunk(passage, observation, {**metadata, PASSAGE: True})
                for passage in passages(facts, size)
            )
    return chunks


# A line that ends a sentence (or introduces what follows) is not wrapped into the next.
_ENDS = (".", "!", "?", ":", ";")
# A line that is a link starts in lower case but continues nothing.
_LINK = re.compile(r"[a-z][a-z0-9+.-]*://")


def _page_lines(text: str) -> list[str]:
    """A page's lines as sentences: wrapped lines rejoined, markdown marks taken off.

    Page text keeps the page's line breaks, so "I built RoboSathi because I could not find a
    single free resource" / "that taught AI from genuine first principles" arrived as two
    premises, the second about no one. A line continues the one before when that one ends no
    sentence and this one starts in lower case. "### Deep Learning Fundamentals" and "Deep
    Learning Fundamentals" are one line once the heading mark is off (and read once).
    """
    lines: list[str] = []
    for raw in text.splitlines():
        line = plain_text(raw)
        continues = line[:1].islower() and not _LINK.match(line)
        if lines and lines[-1] and continues and not lines[-1].endswith(_ENDS):
            lines[-1] = f"{lines[-1]} {line}"
        else:
            lines.append(line)
    return lines


def _chunk(content: str, observation: Observation, metadata: dict[str, object]) -> Chunk:
    return Chunk(
        content=content, kind=TOOL_RESULT_KIND, source=observation.source, metadata=metadata
    )


def passages(facts: list[str], size: int) -> list[str]:
    """Every run of `size` consecutive facts, one string each; none when there are too few.

    A result's facts here; an agent's policy article's sentences, section by section, too.

    A refused call's marker line stays alone: what was refused is not joined to what was read.
    """
    usable = [fact for fact in facts if not fact.startswith(NOT_EXECUTED_MARKER)]
    return [" ".join(usable[start : start + size]) for start in range(len(usable) - size + 1)]


def observation_facts(line: str) -> list[str]:
    """One observation line as the single facts NLI can read, one premise each."""
    if line.startswith(NOT_EXECUTED_MARKER):
        return [line]
    return [sentence for sentence in split_sentences(line) if sentence.strip()]


__all__ = ["TOOL_RESULT_KIND", "Observation", "observation_facts", "passages", "tool_evidence"]
