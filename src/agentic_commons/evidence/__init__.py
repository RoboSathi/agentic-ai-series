"""Append-only evidence with a deterministic verbatim-span rule (a generic mechanism)."""

from agentic_commons.evidence.ledger import EvidenceLedger, RejectionObserver
from agentic_commons.evidence.types import (
    EvidenceCandidate,
    EvidenceEntry,
    LedgerSnapshot,
    RejectionCode,
    SpanRejection,
)
from agentic_commons.evidence.validator import (
    CandidateValidator,
    NonEmptySpanValidator,
    SourceTextLookup,
    VerbatimSpanValidator,
)

__all__ = [
    "CandidateValidator",
    "NonEmptySpanValidator",
    "EvidenceCandidate",
    "EvidenceEntry",
    "EvidenceLedger",
    "LedgerSnapshot",
    "RejectionCode",
    "RejectionObserver",
    "SourceTextLookup",
    "SpanRejection",
    "VerbatimSpanValidator",
]
