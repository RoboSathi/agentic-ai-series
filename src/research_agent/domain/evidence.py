"""What the quarantined extractor may return: claim candidates, nothing else.

The schema *is* the quarantine. The extractor is the only component that
reads raw page text; whatever an injected page persuades it to say, the
only thing that leaves is a list of `{claim, span, url, page_hash,
retrieved_at}` — and the span is then checked verbatim against the page.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ClaimCandidate(BaseModel):
    """One claim the extractor found, with the exact words that support it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str
    span: str
    url: str
    page_hash: str
    retrieved_at: datetime


class ExtractionResult(BaseModel):
    """The extractor's whole output for one page."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[ClaimCandidate] = Field(default_factory=list)
