"""What every context-failure check shares: how a finding is worded, how a chunk splits into
sentences, and the local NLI model the prose checks may ask."""

import re
from typing import Any, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.context.types import (
    Chunk,
    ContextFailureMode,
    ContextFinding,
    is_provenance_header,
)
from agent_platform.scoring.entailment import EntailmentResult

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# What a statement is *about*: identifier-shaped tokens (an order number, a ticket id).
_ENTITY = re.compile(r"\b(?=[\w-]*\d)[A-Za-z0-9][\w-]{3,}\b")


@runtime_checkable
class ContradictionModel(Protocol):
    """An NLI model (`EntailmentModel`): does the premise contradict the hypothesis?"""

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult: ...


def finding(
    mode: ContextFailureMode,
    template: str,
    *,
    source: str,
    action: str = "flagged",
    index: int | None = None,
    excise: list[str] | None = None,
    statements: list[str] | None = None,
    line: str = "",
    **values: Any,
) -> ContextFinding:
    """A finding worded from `context.findings.templates[template]`, its action from `actions`."""
    words = get_settings().context.findings
    lines = words.templates[template]
    return ContextFinding(
        mode=mode,
        source=source,
        rule=lines.rule.format(**values),
        detail=lines.detail.format(**values),
        action=words.actions[action],
        index=index,
        line=line,
        excise=excise or [],
        statements=statements or [],
    )


def snippet(text: str) -> str:
    """The start of `text` on one line, as long as a card line may be."""
    flat = " ".join(text.split())
    limit = get_settings().context.window.snippet_chars
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def label(chunk: Chunk) -> str:
    return chunk.source or chunk.kind


def body(chunk: Chunk) -> str:
    """A chunk's text without its provenance label."""
    return "\n".join(line for line in chunk.content.splitlines() if not is_provenance_header(line))


def sentences(text: str) -> list[str]:
    """The text's sentences long enough to state something (`min_sentence_chars`)."""
    floor = get_settings().context.failure_detection.min_sentence_chars
    return [part.strip() for part in _SENTENCE.split(text) if len(part.strip()) >= floor]


def entities(text: str) -> set[str]:
    """The identifiers a statement names (case-folded)."""
    return {match.casefold() for match in _ENTITY.findall(text)}


def same_subject(left: str, right: str) -> bool:
    """Whether two statements are about the same thing: they name a common identifier.

    NLI reads any two sentences as premise and hypothesis, so "order 10040 was refunded"
    against "order 10790 is cancelled", or a general rule against one order's record, comes
    back as a contradiction. Only statements naming the same record can disagree about it.
    """
    return bool(entities(left) & entities(right))


def contradicts(result: EntailmentResult) -> bool:
    """A calibrated contradiction at or above the configured score (a heuristic never counts)."""
    floor = get_settings().context.failure_detection.contradiction_score
    return result.calibrated and result.label == "contradiction" and result.score >= floor
