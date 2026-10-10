"""The worker end to end over fakes: Tavily-style RFC 2822 dates must reach the fetch step."""

from datetime import datetime, timezone

import pytest

from agent_platform.evidence import EvidenceLedger, NonEmptySpanValidator
from agent_platform.loop_control import BudgetGuard
from agent_platform.web import FetchedPage, SearchHit, SearchRequest, SearchResponse
from research_agent.config import get_agent_settings
from research_agent.domain.budget import Budget
from research_agent.domain.evidence import ClaimCandidate
from research_agent.domain.plan import SubQuestion
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import ScopeBrief
from research_agent.research.extractor import PageReading
from research_agent.research.resources import RunResources
from research_agent.research.worker import ResearcherWorker

pytestmark = pytest.mark.usefixtures("ra_config")

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
BUDGET = Budget(
    max_searches=3, max_fetches=8, max_tokens=80_000, max_seconds=300,
    max_rounds=2, max_dag_nodes=2, max_gap_fill_rounds=0, concurrency=2,
)  # fmt: skip


class _Telemetry:
    def stage(self, *args: object, **kwargs: object) -> None: ...
    def card(self, *args: object, **kwargs: object) -> None: ...
    def audit(self, *args: object, **kwargs: object) -> None: ...


class _Gateway:
    """A search that answers with fixed hits, and pages that are always readable."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.requests: list[SearchRequest] = []
        self.fetched: list[str] = []
        self._pages: dict[str, FetchedPage] = {}

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        return SearchResponse(request=request, hits=self._hits, provider="tavily")

    def page_for(self, url: str) -> FetchedPage | None:
        return self._pages.get(url)

    async def fetch(self, url: str) -> FetchedPage:
        self.fetched.append(url)
        page = FetchedPage(url=url, status=200, text=f"News from {url}.", page_hash="h" + url[-4:])
        self._pages[url] = page
        return page


class _Reader:
    async def read(self, question: str, page: FetchedPage) -> PageReading:
        claim = ClaimCandidate(
            claim=f"Something happened ({page.url})",
            span=page.text,
            url=page.url,
            page_hash=page.page_hash,
            retrieved_at=page.retrieved_at,
        )
        return PageReading(candidates=[claim])


def _worker(hits: list[SearchHit]) -> tuple[ResearcherWorker, _Gateway, RunResources]:
    brief = ScopeBrief(
        archetype="fact_verification", topic="AI news", recency_days=30, budget=BUDGET
    )
    contract = Contract.freeze("run-test", brief, Rubric(), BUDGET).model_copy(
        update={"frozen_at": NOW}
    )
    gateway = _Gateway(hits)
    resources = RunResources(
        contract=contract,
        gateway=gateway,  # type: ignore[arg-type]
        ledger=EvidenceLedger(NonEmptySpanValidator(), "E"),
        budget=BudgetGuard(BUDGET.counters(), max_seconds=BUDGET.max_seconds),
        reader=_Reader(),
        telemetry=_Telemetry(),  # type: ignore[arg-type]
        settings=get_agent_settings().research,
    )
    return ResearcherWorker(resources), gateway, resources


NODE = SubQuestion(node_id="n1", question="AI news?", rubric_items=("R1",), queries=("ai news",))


@pytest.mark.asyncio
async def test_rfc2822_dated_hits_are_fetched_and_admitted() -> None:
    hits = [
        SearchHit(url="https://a.example/one", published_date="Tue, 29 Sep 2026 13:53:01 GMT"),
        SearchHit(url="https://b.example/two", published_date="Sat, 12 Sep 2026 11:00:00 GMT"),
    ]
    worker, gateway, resources = _worker(hits)
    result = await worker.run(NODE)
    assert gateway.fetched == ["https://a.example/one", "https://b.example/two"]
    assert len(result.entries) == 2
    assert not resources.trace.rejected_sources
    assert gateway.requests[0].recent_days == 30 and gateway.requests[0].topic == "news"


@pytest.mark.asyncio
async def test_stale_and_unreadable_dates_are_rejected_with_distinct_codes() -> None:
    hits = [
        SearchHit(url="https://a.example/old", published_date="Mon, 01 Jan 2024 00:00:00 GMT"),
        SearchHit(url="https://a.example/blank", published_date=""),
        SearchHit(url="https://a.example/new", published_date="2026-09-30"),
    ]
    worker, gateway, resources = _worker(hits)
    resources.settings = resources.settings.model_copy(update={"undated_policy": "reject"})
    await worker.run(NODE)
    assert gateway.fetched == ["https://a.example/new"]
    codes = resources.trace.rejected_codes
    assert codes == {"https://a.example/old": "recency", "https://a.example/blank": "undated"}
    assert (
        "no readable publication date"
        in resources.trace.rejected_sources["https://a.example/blank"]
    )
