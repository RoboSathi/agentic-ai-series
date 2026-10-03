"""The verbatim rule: a span is evidence only if the source contains it, exactly.

Deterministic and deliberately unforgiving. No model is asked whether a
paraphrase is "close enough"; the span must be a substring of the text
that was actually retrieved, and that text must be the text whose hash
the candidate claims. A component that invents a quotation — or quotes a
page other than the one it fetched — is caught here by string equality,
which no phrasing can talk its way past.
"""

from typing import Protocol, runtime_checkable

from agent_platform.evidence.types import EvidenceCandidate, SpanRejection
from agent_platform.primitives.hashing import text_hash


@runtime_checkable
class SourceTextLookup(Protocol):
    """Returns the exact text retrieved for a source, or None if never retrieved."""

    def text_for(self, source_url: str) -> str | None: ...


@runtime_checkable
class CandidateValidator(Protocol):
    """Decides whether a candidate may enter the ledger (None = admit)."""

    def check(self, candidate: EvidenceCandidate) -> SpanRejection | None: ...


class NonEmptySpanValidator:
    """Admits any candidate with a span — the rule a ledger falls back to
    when verbatim checking is deliberately withheld (an ablation), so the
    difference the verbatim rule makes can be measured."""

    def check(self, candidate: EvidenceCandidate) -> SpanRejection | None:
        if not candidate.span.strip():
            return SpanRejection(candidate=candidate, reason_code="EMPTY_SPAN")
        return None


class VerbatimSpanValidator:
    """Admits a candidate only if its span is an exact substring of its source."""

    def __init__(self, sources: SourceTextLookup) -> None:
        self._sources = sources

    def check(self, candidate: EvidenceCandidate) -> SpanRejection | None:
        """None when admissible, otherwise the reason it is not."""
        if not candidate.span.strip():
            return SpanRejection(candidate=candidate, reason_code="EMPTY_SPAN")
        text = self._sources.text_for(candidate.source_url)
        if text is None:
            return SpanRejection(
                candidate=candidate,
                reason_code="SOURCE_UNKNOWN",
                detail="source was never retrieved",
            )
        if text_hash(text) != candidate.source_hash:
            return SpanRejection(
                candidate=candidate,
                reason_code="SOURCE_HASH_MISMATCH",
                detail="span cites a different version of the source",
            )
        if candidate.span not in text:
            return SpanRejection(
                candidate=candidate,
                reason_code="SPAN_NOT_VERBATIM",
                detail="span is not an exact substring of the retrieved text",
            )
        return None
