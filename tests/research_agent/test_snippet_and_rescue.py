"""A run never ends with nothing to show (run-3330c3dcdd: 3 hits outside the window, 2 paywalled).

A page that could not be read leaves its snippet as labelled evidence; a ledger still empty at
reflection gets one rescue round without the time window; still empty at synthesis, the search
results themselves are the evidence of a best-effort report. Every such entry is caveated.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.evidence import EvidenceLedger, VerbatimSpanValidator
from agent_platform.loop_control import BudgetGuard
from agent_platform.primitives.hashing import text_hash
from agent_platform.web import FetchedPage, SearchHit, SearchRequest, SearchResponse
from research_agent.config import get_agent_settings
from research_agent.domain.budget import FETCHES, ROUNDS, SEARCHES, Budget
from research_agent.domain.evidence import ClaimCandidate
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import ScopeBrief
from research_agent.domain.trace import SeenHit
from research_agent.graph.run_nodes import RunNodes
from research_agent.research.extractor import PageReading
from research_agent.research.resources import RunResources
from research_agent.research.snippets import SnippetSources
from research_agent.research.worker import ResearcherWorker

pytestmark = pytest.mark.usefixtures("ra_config")

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
BUDGET = Budget(
    max_searches=1, max_fetches=4, max_tokens=80_000, max_seconds=300,
    max_rounds=1, max_dag_nodes=2, max_gap_fill_rounds=0, concurrency=2,
)  # fmt: skip
SNIPPET = (
    "Loop control in agentic systems bounds each run with a step cap and a token budget. Short."
)
NODE = SubQuestion(
    node_id="Q1", question="How is loop control done?", rubric_items=("R1",),
    queries=("agentic loop control",),
)  # fmt: skip


class _Telemetry:
    reporter = None

    def __init__(self) -> None:
        self.cards: list[tuple[str, str, Any, str]] = []
        self.audits: list[str] = []

    def stage(self, *args: object, **kwargs: object) -> None: ...

    def card(self, stage: str, heading: str, sections: Any, status: str = "completed") -> None:
        self.cards.append((stage, heading, sections, status))

    def audit(self, event: str, *args: object, **kwargs: object) -> None:
        self.audits.append(event)


class _Web:
    """Fixed hits; each URL fetches to a status and text."""

    def __init__(self, hits: list[SearchHit], pages: dict[str, tuple[int, str]]) -> None:
        self._hits, self._pages = hits, pages
        self.requests: list[SearchRequest] = []
        self._fetched: dict[str, FetchedPage] = {}

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        return SearchResponse(request=request, hits=self._hits, provider="tavily")

    def page_for(self, url: str) -> FetchedPage | None:
        return self._fetched.get(url)

    def text_for(self, url: str) -> str | None:
        page = self._fetched.get(url)
        return page.text if page else None

    async def fetch(self, url: str) -> FetchedPage:
        status, text = self._pages[url]
        page = FetchedPage(url=url, status=status, text=text, page_hash=text_hash(text))
        self._fetched[url] = page
        return page


class _Reader:
    """Quotes each readable page whole."""

    async def read(self, question: str, page: FetchedPage) -> PageReading:
        claim = ClaimCandidate(
            claim="A finding", span=page.text, url=page.url, page_hash=page.page_hash,
            retrieved_at=page.retrieved_at,
        )  # fmt: skip
        return PageReading(candidates=[claim])


def _resources(hits: list[SearchHit], pages: dict[str, tuple[int, str]]) -> RunResources:
    brief = ScopeBrief(archetype="general", topic="loop control", recency_days=90, budget=BUDGET)
    contract = Contract.freeze("run-test", brief, Rubric(), BUDGET).model_copy(
        update={"frozen_at": NOW}
    )
    web = _Web(hits, pages)
    snippets = SnippetSources(web)
    return RunResources(
        contract=contract,
        gateway=web,  # type: ignore[arg-type]
        ledger=EvidenceLedger(VerbatimSpanValidator(snippets), "E"),
        budget=BudgetGuard(BUDGET.counters(), max_seconds=300),
        reader=_Reader(),
        telemetry=_Telemetry(),  # type: ignore[arg-type]
        settings=get_agent_settings().research,
        snippets=snippets,
    )


class TestASnippetStandsInForAPageThatCouldNotBeRead:
    async def test_a_paywalled_page_leaves_its_snippet_as_caveated_evidence(self) -> None:
        url = "https://paywalled.example/loop-control"
        hit = SearchHit(url=url, snippet=SNIPPET, published_date="2026-09-20")
        resources = _resources([hit], {url: (403, "")})
        result = await ResearcherWorker(resources).run(NODE)
        entries = resources.ledger.snapshot().entries
        caveat = resources.settings.snippet_evidence.caveat
        # The short sentence is not a claim; the long one is, verbatim from the snippet.
        assert [e.claim for e in entries] == [SNIPPET.split(" Short.")[0]]
        assert entries[0].caveat == caveat and entries[0].source_url == url
        assert result.entries == [entries[0].entry_id]

    async def test_a_page_withheld_for_injection_is_never_cited_through_its_snippet(
        self,
    ) -> None:
        url = "https://hostile.example/page"
        resources = _resources(
            [SearchHit(url=url, snippet=SNIPPET, published_date="2026-09-20")], {}
        )
        resources.trace.reject(url, "injection detected", "injection")
        worker = ResearcherWorker(resources)
        worker._from_snippet(NODE, SearchHit(url=url, snippet=SNIPPET), _result())
        assert not resources.ledger.snapshot().entries


def _result() -> Any:
    from research_agent.research.worker import NodeResult

    return NodeResult(node_id=NODE.node_id)


class TestTheRescueRoundRelaxesTheWindow:
    async def test_out_of_window_sources_are_admitted_with_a_caveat_once_rescued(self) -> None:
        url = "https://old.example/loop"
        hit = SearchHit(url=url, snippet="", published_date="2025-01-10")
        resources = _resources([hit], {url: (200, "Loop control caps every agent run.")})
        resources.trace.rescued = True
        await ResearcherWorker(resources).run(NODE)
        entries = resources.ledger.snapshot().entries
        assert len(entries) == 1 and "outside the approved 90-day window" in entries[0].caveat
        # The provider is asked for any date in the rescue round.
        web: Any = resources.gateway
        assert web.requests[0].recent_days is None

    async def test_before_the_rescue_the_window_still_rejects(self) -> None:
        url = "https://old.example/loop"
        hit = SearchHit(url=url, snippet="", published_date="2025-01-10")
        resources = _resources([hit], {url: (200, "Loop control caps every agent run.")})
        await ResearcherWorker(resources).run(NODE)
        assert not resources.ledger.snapshot().entries
        assert resources.trace.rejected_codes == {url: "recency"}


def _nodes(resources: RunResources) -> RunNodes:
    parts: Any = SimpleNamespace(node_id_prefix="Q", clip_chars=200)
    nodes = RunNodes(parts)
    nodes.bind(resources)
    return nodes


def _state(resources: RunResources) -> Any:
    return {
        "contract": resources.contract,
        "plans": [PlanDag(nodes=(NODE,), rationale="plan")],
        "round": 1,
    }


class TestARunThatAdmittedNothingGetsOneRescueRound:
    def test_the_plan_runs_again_under_a_grant_and_only_once(self) -> None:
        resources = _resources([], {})
        resources.budget.charge(ROUNDS, 1)
        resources.budget.charge(SEARCHES, 1)
        nodes = _nodes(resources)
        update = nodes._rescue(_state(resources))
        assert update is not None
        assert [n.question for n in update["round_nodes"]] == [NODE.question]
        assert update["round_nodes"][0].node_id == "Q2"
        rescue = resources.settings.rescue
        assert resources.budget.remaining(SEARCHES) == rescue.searches
        assert resources.budget.snapshot()[FETCHES].cap == BUDGET.max_fetches + rescue.fetches
        assert resources.trace.rescued and "scope.recency_relaxed" in _audits(resources)
        assert nodes._rescue(_state(resources)) is None

    def test_a_run_with_evidence_needs_no_rescue(self) -> None:
        url = "https://paywalled.example/x"
        resources = _resources([], {})
        assert resources.snippets is not None
        page_hash = resources.snippets.remember(url, SNIPPET)
        from agent_platform.evidence import EvidenceCandidate

        resources.ledger.append(
            EvidenceCandidate(claim="c", span=SNIPPET, source_url=url, source_hash=page_hash)
        )
        assert _nodes(resources)._rescue(_state(resources)) is None


class TestAStillEmptyLedgerIsWrittenUpFromSearchResults:
    def test_the_search_results_become_caveated_entries_and_the_report_says_so(self) -> None:
        resources = _resources([], {})
        resources.trace.saw(
            SeenHit(node_id="Q1", url="https://a.example/x", snippet=SNIPPET, title="A")
        )
        notice = _nodes(resources)._best_effort()
        rescue = resources.settings.rescue
        assert notice == rescue.best_effort_note
        entries = resources.ledger.snapshot().entries
        assert entries and all(e.caveat == rescue.best_effort_caveat for e in entries)
        assert _nodes(resources)._rests_on_weaker_evidence()

    def test_no_results_at_all_leaves_no_notice(self) -> None:
        resources = _resources([], {})
        assert _nodes(resources)._best_effort() == ""


def _audits(resources: RunResources) -> list[str]:
    telemetry: Any = resources.telemetry
    return list(telemetry.audits)
