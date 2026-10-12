"""Evidence value types: a claim, where it came from, and the exact words.

An entry says *this claim* is supported by *this span* of *this source*,
retrieved *then*, whose full text hashed to *this*. Every field is there
so the support can be re-checked later by someone who trusts none of the
components that produced it.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.observability.types import utc_now

RejectionCode = Literal["EMPTY_SPAN", "SOURCE_UNKNOWN", "SOURCE_HASH_MISMATCH", "SPAN_NOT_VERBATIM"]


class EvidenceCandidate(BaseModel):
    """A proposed entry, before the verbatim rule has admitted it."""

    model_config = ConfigDict(frozen=True)

    claim: str
    span: str
    source_url: str
    source_hash: str
    retrieved_at: datetime = Field(default_factory=utc_now)
    # When the source says it was published (ISO date, "" when unknown).
    source_date: str = ""
    # What the claim is evidence *for* — an opaque key the caller chooses.
    subject: str = ""
    # Why this entry is weaker than a span of a page read in full and in scope ("search
    # snippet only", "outside the approved window"); "" for an ordinary entry. The reader and
    # the writer of a report are told; the verbatim rule applies all the same.
    caveat: str = ""


class EvidenceEntry(EvidenceCandidate):
    """An admitted entry, with the id citations refer to."""

    entry_id: str


class SpanRejection(BaseModel):
    """Why a candidate was refused entry."""

    model_config = ConfigDict(frozen=True)

    candidate: EvidenceCandidate
    reason_code: RejectionCode
    detail: str = ""


class LedgerSnapshot(BaseModel):
    """An immutable copy of a ledger at one moment — what an auditor is handed."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[EvidenceEntry, ...] = ()
    rejections: tuple[SpanRejection, ...] = ()

    def entry(self, entry_id: str) -> EvidenceEntry | None:
        return next((entry for entry in self.entries if entry.entry_id == entry_id), None)

    def for_subject(self, subject: str) -> list[EvidenceEntry]:
        return [entry for entry in self.entries if entry.subject == subject]

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(entry.source_url for entry in self.entries))
