"""Turning a fetched page into claim candidates.

`QuarantinedExtractor` (layer 3) is the only component that reads raw
page text. It stamps each candidate with the URL, hash and retrieval time
of the page it actually fetched — never the model's own account of them —
so a candidate cannot cite a page other than the one it came from.

`UnquarantinedReader` is what runs when the quarantine is withheld: the
researcher role reads the page and may also propose follow-up actions.
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.context.excerpt import relevant_excerpt
from agent_platform.logging import instrument
from agent_platform.observability.step_card import ModelCall
from agent_platform.security.prompt_injection.detector import InstructionDataSeparator
from agent_platform.web import FetchedPage
from research_agent.domain.evidence import ClaimCandidate
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import extractor as extractor_prompt
from research_agent.prompts import researcher as researcher_prompt
from research_agent.prompts.extractor import ExtractedClaim
from research_agent.research.defenses import delimit


class PageReading(BaseModel):
    """What reading one page produced."""

    candidates: list[ClaimCandidate] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    follow_up_urls: list[str] = Field(default_factory=list)
    # The model call that read the page, drawn inside the page's card (one card per page).
    call: ModelCall | None = None


def _stamp(claims: list[ExtractedClaim], page: FetchedPage, limit: int) -> list[ClaimCandidate]:
    retrieved: datetime = page.retrieved_at
    return [
        ClaimCandidate(
            claim=claim.claim,
            span=claim.span,
            url=page.url,
            page_hash=page.page_hash,
            retrieved_at=retrieved,
        )
        for claim in claims[:limit]
    ]


@instrument("research_agent")
class QuarantinedExtractor:
    """A separate call, strict schema, delimited untrusted input."""

    def __init__(
        self,
        caller: StructuredCaller,
        separator: InstructionDataSeparator,
        max_claims: int,
        page_chars: int,
        window_chars: int,
        today: date,
    ) -> None:
        self._caller = caller
        self._today = today
        self._separator = separator
        self._max = max_claims
        self._page_chars = page_chars
        self._window_chars = window_chars

    async def read(self, question: str, page: FetchedPage) -> PageReading:
        text = relevant_excerpt(page.text, question, self._page_chars, self._window_chars)
        wrapped = delimit(self._separator, extractor_prompt.INSTRUCTIONS, text)
        answer, call = await self._caller.call_traced(
            extractor_prompt.build(question, page.url, wrapped, self._max, self._today),
            extractor_prompt.ExtractorAnswer,
        )
        return PageReading(candidates=_stamp(answer.claims, page, self._max), call=call)


class UnquarantinedReader:
    """The researcher reads the page itself and may propose actions (ablation)."""

    def __init__(
        self,
        caller: StructuredCaller,
        max_claims: int,
        brief: dict[str, Any],
        page_chars: int,
        window_chars: int,
        today: date,
    ) -> None:
        self._caller = caller
        self._today = today
        self._max = max_claims
        self._brief = brief
        self._page_chars = page_chars
        self._window_chars = window_chars

    async def read(self, question: str, page: FetchedPage) -> PageReading:
        text = relevant_excerpt(page.text, question, self._page_chars, self._window_chars)
        answer, call = await self._caller.call_traced(
            researcher_prompt.build(question, page.url, text, self._max, self._brief, self._today),
            researcher_prompt.ResearcherAnswer,
        )
        return PageReading(
            candidates=_stamp(answer.claims, page, self._max),
            follow_up_queries=answer.follow_up_queries,
            follow_up_urls=answer.follow_up_urls,
            call=call,
        )
