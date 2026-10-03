"""Append-only evidence with a deterministic verbatim-span rule (a generic mechanism)."""

from agent_platform.evidence.ledger import EvidenceLedger, RejectionObserver
from agent_platform.evidence.types import (
    EvidenceCandidate,
    EvidenceEntry,
    LedgerSnapshot,
    RejectionCode,
    SpanRejection,
)
from agent_platform.evidence.validator import (
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
