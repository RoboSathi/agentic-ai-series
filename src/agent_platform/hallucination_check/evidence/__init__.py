"""Evidence: what a claim may be grounded on.

Two mechanisms share the word:

- `observations` — tool results as grounding premises, one fact per premise,
  in the order they happened (every agent that grounds a reply on its tools).
- `ledger` / `validator` — append-only evidence with a deterministic
  verbatim-span rule (a quoted span must occur in its source text).
"""

from agent_platform.hallucination_check.evidence.ledger import EvidenceLedger, RejectionObserver
from agent_platform.hallucination_check.evidence.observations import (
    TOOL_RESULT_KIND,
    Observation,
    observation_facts,
    passages,
    tool_evidence,
)
from agent_platform.hallucination_check.evidence.types import (
    EvidenceCandidate,
    EvidenceEntry,
    LedgerSnapshot,
    RejectionCode,
    SpanRejection,
)
from agent_platform.hallucination_check.evidence.validator import (
    CandidateValidator,
    NonEmptySpanValidator,
    SourceTextLookup,
    VerbatimSpanValidator,
)

__all__ = [
    "TOOL_RESULT_KIND",
    "CandidateValidator",
    "EvidenceCandidate",
    "EvidenceEntry",
    "EvidenceLedger",
    "LedgerSnapshot",
    "NonEmptySpanValidator",
    "Observation",
    "RejectionCode",
    "RejectionObserver",
    "SourceTextLookup",
    "SpanRejection",
    "VerbatimSpanValidator",
    "observation_facts",
    "passages",
    "tool_evidence",
]
