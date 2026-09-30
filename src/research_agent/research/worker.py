"""The researcher worker: one plan node, a bounded search → fetch → read loop.

Per query: outbound check → reserve a search → search → source policy →
for each hit: outbound check → reserve a fetch (cached pages are free) →
fetch → content defence → read (quarantined extractor, or the researcher
itself when the quarantine is withheld) → ledger append under the
verbatim rule.

The worker holds two capabilities and no others: `search` and `fetch`.
There is no tool that writes, sends or pays for anything, so the worst a
hostile page can make a worker do is look something up — which is why
the outbound check on what it looks up matters.

Budget is reserved *before* each awaited call and released if the call
fails, so parallel workers cannot overshoot a cap. The loop is bounded
by the node's queries (plus any follow-ups) and by the per-node fetch cap.
"""

from pydantic import BaseModel, Field

from agentic_commons.evidence import EvidenceCandidate, EvidenceEntry, SpanRejection
from agentic_commons.resilience import QuotaExhaustedError
from agentic_commons.web import FetchedPage, SearchHit, SearchRequest, SearchResponse
from research_agent.domain.budget import FETCHES, SEARCHES
from research_agent.domain.evidence import ClaimCandidate
from research_agent.domain.plan import SubQuestion
from research_agent.research.defenses import apply_source_policy, within_window
from research_agent.research.resources import RunResources


class NodeResult(BaseModel):
    """What one node's research produced."""

    node_id: str
    queries: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    entries: list[str] = Field(default_factory=list)
    stopped: str = ""


class ResearcherWorker:
    """Researches one sub-question with the run's shared resources."""

    def __init__(self, resources: RunResources) -> None:
        self._r = resources

    async def run(self, node: SubQuestion, extra_queries: tuple[str, ...] = ()) -> NodeResult:
        result = NodeResult(node_id=node.node_id)
        queue = [*node.queries, *extra_queries]
        while queue and not result.stopped:
            if self._r.cancelled:
                result.stopped = "cancelled"
                break
            await self._query(node, queue.pop(0), queue, result)
        return result

    # -- search ---------------------------------------------------------------------

    async def _query(
        self, node: SubQuestion, query: str, queue: list[str], result: NodeResult
    ) -> None:
        if not self._outbound_ok("query", query):
            return
        response = await self._search(query, node, result)
        if response is None:
            return
        hits = self._policy(response.hits)[: self._r.settings.max_results_per_search]
        for hit in hits:
            if len(result.sources) >= self._r.settings.max_fetches_per_node or result.stopped:
                return
            await self._visit(node, hit, queue, result)

    async def _search(
        self, query: str, node: SubQuestion, result: NodeResult
    ) -> SearchResponse | None:
        held = self._r.budget.try_reserve(SEARCHES)
        if held is None:
            result.stopped = "budget"
            self._r.trace.budget_stops.append(f"{node.node_id}: searches exhausted")
            return None
        request = self._request(query)
        self._r.trace.attempted_outbound.append(query)
        try:
            response = await self._r.gateway.search(request)
        except Exception as exc:
            self._r.budget.release(held)
            self._failed("worker_search", query, exc)
            if _quota_spent(exc):
                result.stopped = "budget"
                self._r.trace.budget_stops.append(self._r.settings.quota_stop)
            return None
        self._r.budget.commit(held)
        self._searched(node, query, response, result)
        return response

    def _request(self, query: str) -> SearchRequest:
        """The approved time window goes to the provider; a short one searches news."""
        days = self._r.contract.brief.recency_days
        news = days is not None and days <= self._r.settings.news_window_days
        return SearchRequest(
            query=query,
            max_results=self._r.settings.max_results_per_search,
            recent_days=days,
            topic=self._r.settings.news_topic if news else "",
        )

    def _searched(
        self, node: SubQuestion, query: str, response: SearchResponse, result: NodeResult
    ) -> None:
        self._r.trace.tried(node.node_id, query)
        result.queries.append(query)
        if response.provider_summary:
            # Carried for display only; it never becomes a page or an entry.
            self._r.trace.provider_summaries_seen.append(response.provider_summary)
        self._r.telemetry.audit(
            "web.search", "allow", query, node=node.node_id, hits=len(response.hits)
        )
        self._r.telemetry.stage(
            "worker_search",
            f"{node.node_id} · '{query}' → {len(response.hits)} hits",
            {
                "node": node.node_id,
                "query": query,
                "hits": len(response.hits),
                "provider_summary": "ignored (untrusted)" if response.provider_summary else "",
            },
        )

    def _policy(self, hits: list[SearchHit]) -> list[SearchHit]:
        in_scope = [hit for hit in hits if self._recent(hit)]
        if self._r.outbound is None:
            return in_scope
        return apply_source_policy(in_scope, self._r.contract.brief)

    def _recent(self, hit: SearchHit) -> bool:
        """The approved recency window is scope, not a defence: it always applies."""
        contract = self._r.contract
        days = contract.brief.recency_days
        if days is None or within_window(hit.published_date, contract.frozen_at.date(), days):
            return True
        reason = self._r.settings.recency_rejection.format(days=days, date=hit.published_date)
        self._r.trace.reject(hit.url, reason)
        return False

    # -- fetch and read -------------------------------------------------------------

    async def _visit(
        self, node: SubQuestion, hit: SearchHit, queue: list[str], result: NodeResult
    ) -> None:
        page = await self._fetch(hit.url, result)
        if page is None:
            return
        result.sources.append(page.url)
        self._r.trace.examined(page.url)
        if not self._clean(page):
            return
        reading = await self._r.reader.read(node.question, page)
        self._admit(node, reading.candidates, hit, result)
        await self._follow(node, reading.follow_up_queries, reading.follow_up_urls, queue, result)

    async def _fetch(self, url: str, result: NodeResult) -> FetchedPage | None:
        cached = self._r.gateway.page_for(url)
        if cached is not None:
            return cached if cached.ok else None
        if not self._outbound_ok("url", url):
            return None
        return await self._fetch_new(url, result)

    async def _fetch_new(self, url: str, result: NodeResult) -> FetchedPage | None:
        held = self._r.budget.try_reserve(FETCHES)
        if held is None:
            result.stopped = "budget"
            self._r.trace.budget_stops.append(f"{result.node_id}: fetches exhausted")
            return None
        self._r.trace.attempted_outbound.append(url)
        try:
            page = await self._r.gateway.fetch(url)
        except Exception as exc:
            self._r.budget.release(held)
            self._failed("worker_fetch", url, exc)
            return None
        self._r.budget.commit(held)
        return self._fetched(page)

    def _fetched(self, page: FetchedPage) -> FetchedPage | None:
        self._r.telemetry.audit(
            "web.fetch", "allow" if page.ok else "deny", page.url, status=page.status, via=page.via
        )
        self._r.telemetry.stage(
            "worker_fetch",
            f"{page.url} · {page.status} · {len(page.text)} chars"
            + (" · via reader" if page.via == "reader" else ""),
            {
                "url": page.url,
                "status": page.status,
                "via": page.via,
                "page_hash": page.page_hash[:12],
            },
            status="completed" if page.ok else "warning",
        )
        if not page.ok:
            self._r.trace.examined(page.url)
            self._r.trace.reject(page.url, f"dead link (HTTP {page.status})")
            return None
        return page

    def _clean(self, page: FetchedPage) -> bool:
        """Layer 2: drop a page the injection detector flags."""
        if self._r.screen is None:
            return True
        phrases = self._r.screen.flagged(page.text)
        if not phrases:
            return True
        self._r.trace.reject(page.url, f"injection detected: {', '.join(phrases)}")
        self._r.telemetry.audit(
            "security.page_rejected",
            "deny",
            "prompt injection in fetched text",
            url=page.url,
            phrases=phrases,
        )
        self._r.telemetry.stage(
            "content_defense",
            f"Page dropped — injection phrasing: {', '.join(phrases)}",
            {"url": page.url, "phrases": phrases, "layer": "classifier"},
            status="blocked",
        )
        return False

    def _admit(
        self,
        node: SubQuestion,
        candidates: list[ClaimCandidate],
        hit: SearchHit,
        result: NodeResult,
    ) -> None:
        for candidate in candidates:
            outcome = self._r.ledger.append(
                EvidenceCandidate(
                    claim=candidate.claim,
                    span=candidate.span,
                    source_url=candidate.url,
                    source_hash=candidate.page_hash,
                    retrieved_at=candidate.retrieved_at,
                    source_date=hit.published_date,
                    subject=node.node_id,
                )
            )
            self._record(outcome, result)
        self._r.telemetry.stage(
            "extract_claims",
            f"{node.node_id} · {len(candidates)} claim(s) from {hit.url}",
            {"node": node.node_id, "url": hit.url, "claims": len(candidates)},
        )

    def _record(self, outcome: EvidenceEntry | SpanRejection, result: NodeResult) -> None:
        if isinstance(outcome, EvidenceEntry):
            result.entries.append(outcome.entry_id)
            self._r.telemetry.stage(
                "ledger_append",
                f"{outcome.entry_id} · {outcome.claim[:70]}",
                {"entry": outcome.entry_id, "url": outcome.source_url},
            )
            return
        self._r.trace.span_rejections += 1
        self._r.telemetry.stage(
            "span_rejected",
            f"Span rejected ({outcome.reason_code})",
            {
                "url": outcome.candidate.source_url,
                "reason_code": outcome.reason_code,
                "span": outcome.candidate.span[:80],
            },
            status="warning",
        )

    async def _follow(
        self,
        node: SubQuestion,
        queries: list[str],
        urls: list[str],
        queue: list[str],
        result: NodeResult,
    ) -> None:
        """Follow-up actions a reader proposed (only the unquarantined reader can)."""
        queue.extend(queries)
        for url in urls:
            await self._visit(node, SearchHit(url=url), queue, result)

    # -- shared ------------------------------------------------------------------------

    def _outbound_ok(self, kind: str, value: str) -> bool:
        if self._r.outbound is None:
            return True
        verdict = (
            self._r.outbound.check_query(value)
            if kind == "query"
            else self._r.outbound.check_url(value)
        )
        if verdict.allowed:
            return True
        self._r.trace.blocked_outbound.append(f"{kind}: {verdict.reason_code}")
        self._r.telemetry.audit(
            "security.outbound_blocked",
            "deny",
            verdict.reason_code,
            kind=kind,
            detail=verdict.detail,
        )
        self._r.telemetry.stage(
            "content_defense",
            f"Outbound {kind} blocked ({verdict.reason_code})",
            {"kind": kind, "reason_code": verdict.reason_code, "layer": "structural"},
            status="blocked",
        )
        return False

    def _failed(self, stage: str, target: str, exc: Exception) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        self._r.trace.errors.append(f"{target}: {detail}")
        self._r.telemetry.audit(stage.replace("worker_", "web."), "deny", detail, target=target)
        self._r.telemetry.stage(
            stage,
            f"{target} failed — {detail}",
            {"target": target, "error": detail},
            status="failed",
        )


def _quota_spent(exc: BaseException | None) -> bool:
    """Whether a search failed because every provider is out of monthly quota."""
    while exc is not None:
        if isinstance(exc, QuotaExhaustedError):
            return True
        exc = exc.__cause__
    return False
