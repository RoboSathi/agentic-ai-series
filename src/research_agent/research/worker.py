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

import asyncio
from datetime import date

from pydantic import BaseModel, Field

from agentic_commons.evidence import EvidenceCandidate, EvidenceEntry
from agentic_commons.logging import describe_error, instrument
from agentic_commons.resilience import QuotaExhaustedError
from agentic_commons.web import FetchedPage, SearchHit, SearchRequest, SearchResponse
from research_agent.config import get_agent_settings
from research_agent.domain.budget import FETCHES, SEARCHES
from research_agent.domain.evidence import ClaimCandidate
from research_agent.domain.plan import SubQuestion
from research_agent.narration import say
from research_agent.research import cards
from research_agent.research.defenses import apply_source_policy, rank_by_kind, window_verdict
from research_agent.research.resources import RunResources
from research_agent.research.spans import canonical_span, nearest_match
from research_agent.structured import OutputTruncatedError
from research_agent.synthesis.reader_view import host_of

# The rejection code of a page the model could not read (its answer was cut off or invalid).
UNREADABLE = "unreadable"


# The bulkhead pool every page read of a run shares (`RunResources.page_reads`).
PAGE_POOL = "page_reads"


class NodeResult(BaseModel):
    """What one node's research produced."""

    node_id: str
    queries: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    entries: list[str] = Field(default_factory=list)
    stopped: str = ""


@instrument("research_agent")
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
        room = self._r.settings.max_fetches_per_node - len(result.sources)
        # The pages are read side by side (fetch, screen, extract), bounded run-wide by the
        # `page_reads` bulkhead; one page's failure never takes the others down.
        await asyncio.gather(*(self._read_in_pool(node, hit, queue, result) for hit in hits[:room]))

    async def _read_in_pool(
        self, node: SubQuestion, hit: SearchHit, queue: list[str], result: NodeResult
    ) -> None:
        async with self._r.page_reads.acquire(PAGE_POOL):
            if result.stopped or self._r.cancelled:
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
        heading, sections = cards.search_card(
            node.node_id,
            query,
            response.provider,
            _window_label(response.request),
            [f"{host_of(h.url)} · {h.published_date[:16] or 'undated'}" for h in response.hits],
            bool(response.provider_summary),
        )
        self._r.telemetry.card("worker_search", heading, sections)

    def _policy(self, hits: list[SearchHit]) -> list[SearchHit]:
        in_scope = [hit for hit in hits if self._recent(hit)]
        if self._r.outbound is not None:
            in_scope = apply_source_policy(in_scope, self._r.contract.brief)
        return rank_by_kind(in_scope, self._r.settings.source_labels)

    def _recent(self, hit: SearchHit) -> bool:
        """The approved recency window is scope, not a defence: it always applies."""
        contract = self._r.contract
        days = contract.brief.recency_days
        if days is None:
            return True
        verdict = window_verdict(hit.published_date, contract.frozen_at.date(), days)
        if verdict == "in" or (verdict == "undated" and self._verify_on_page()):
            return True
        template = (
            self._r.settings.undated_rejection
            if verdict == "undated"
            else self._r.settings.recency_rejection
        )
        reason = template.format(days=days, date=hit.published_date)
        self._r.trace.reject(hit.url, reason, "undated" if verdict == "undated" else "recency")
        return False

    def _verify_on_page(self) -> bool:
        return self._r.settings.undated_policy == "verify_on_page"

    def _page_dated(self, hit: SearchHit, page: FetchedPage) -> SearchHit | None:
        """An undated hit, once read: the page's own date decides, else it is kept unverified.

        The provider already applied the window to its results, so an unreadable date is
        unverified, not out of window; a readable one outside the window is rejected here.
        """
        days = self._r.contract.brief.recency_days
        if days is None or window_verdict(hit.published_date, self._as_of(), days) != "undated":
            return hit
        verdict = window_verdict(page.published, self._as_of(), days)
        if verdict == "out":
            reason = self._r.settings.recency_rejection.format(days=days, date=page.published)
            self._r.trace.reject(hit.url, reason, "recency")
            return None
        return hit.model_copy(update={"published_date": page.published})

    def _as_of(self) -> date:
        return self._r.contract.frozen_at.date()

    # -- fetch and read -------------------------------------------------------------

    async def _visit(
        self, node: SubQuestion, hit: SearchHit, queue: list[str], result: NodeResult
    ) -> None:
        page = await self._fetch(hit.url, node, result)
        if page is None:
            return
        result.sources.append(page.url)
        self._r.trace.examined(page.url)
        dated = self._page_dated(hit, page)
        if dated is None or not await self._clean(page, node):
            return
        hit = dated
        try:
            reading = await self._r.reader.read(node.question, page)
        except Exception as exc:
            # One unreadable page must not take the node's other pages down with it.
            self._unreadable(node, page, exc)
            return
        admitted, refused = self._admit(node, reading.candidates, hit, page, result)
        self._page_card(node, page, len(reading.candidates), admitted, refused)
        await self._follow(node, reading.follow_up_queries, reading.follow_up_urls, queue, result)

    def _unreadable(self, node: SubQuestion, page: FetchedPage, exc: Exception) -> None:
        """A page the model could not turn into claims: recorded, shown, and explained."""
        why = (
            say("page.unreadable_cut_off")
            if isinstance(exc, OutputTruncatedError)
            else say("page.unreadable_error", error=describe_error(exc))
        )
        self._r.trace.errors.append(f"{node.node_id}: {page.url}: {describe_error(exc)}")
        self._r.trace.reject(page.url, why, UNREADABLE)
        self._r.telemetry.audit(
            "web.read", "deny", describe_error(exc), node=node.node_id, url=page.url
        )
        heading, sections = cards.unreadable_card(host_of(page.url), page.url, why, node.node_id)
        self._r.telemetry.card("page_result", heading, sections, status="warning")

    async def _fetch(self, url: str, node: SubQuestion, result: NodeResult) -> FetchedPage | None:
        cached = self._r.gateway.page_for(url)
        if cached is not None:
            return cached if cached.ok else None
        if not self._outbound_ok("url", url):
            return None
        return await self._fetch_new(url, node, result)

    async def _fetch_new(
        self, url: str, node: SubQuestion, result: NodeResult
    ) -> FetchedPage | None:
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
            self._failed("worker_fetch", url, exc, node.node_id)
            self._r.trace.examined(url)
            self._r.trace.reject(url, describe_error(exc), "error")
            return None
        self._r.budget.commit(held)
        return self._fetched(page, node)

    def _fetched(self, page: FetchedPage, node: SubQuestion) -> FetchedPage | None:
        """A readable page goes on to its page card; only a failure draws a card of its own."""
        self._r.telemetry.audit(
            "web.fetch", "allow" if page.ok else "deny", page.url, status=page.status, via=page.via
        )
        if page.ok:
            return page
        code, label = _fetch_outcome(page.status)
        self._r.telemetry.stage(
            "worker_fetch",
            f"{node.node_id} · {host_of(page.url)} · HTTP {page.status} — {label}",
            {
                "node": node.node_id,
                "url": page.url,
                "status": page.status,
                "via": page.via,
                "page_hash": page.page_hash[:12],
            },
            status="warning",
        )
        self._r.trace.examined(page.url)
        self._r.trace.reject(page.url, f"{label} (HTTP {page.status})", code)
        return None

    async def _clean(self, page: FetchedPage, node: SubQuestion) -> bool:
        """Layer 2: the phrase check, with the injection model as a second opinion."""
        if self._r.screen is None:
            return True
        verdict = await self._r.screen.assess(page.text)
        if verdict.decision == "clean":
            return True
        detail = {
            "node": node.node_id,
            "url": page.url,
            "phrases": verdict.phrases,
            "model": verdict.model,
            "model_score": verdict.score,
            "decision": verdict.decision,
            "layer": "classifier+model" if verdict.model else "classifier",
        }
        phrases = ", ".join(verdict.phrases) or "none"
        if not verdict.drop:
            self._r.telemetry.audit(
                "security.page_kept", "allow", "phrase hit cleared by the injection model",
                url=page.url, phrases=verdict.phrases, score=verdict.score,
            )  # fmt: skip
            self._r.telemetry.stage(
                "content_defense",
                f"{node.node_id} · kept {host_of(page.url)} — phrase [{phrases}], {verdict.why}",
                detail,
            )
            return True
        self._r.trace.reject(
            page.url, f"injection detected: {phrases} ({verdict.why})", "injection"
        )
        self._r.telemetry.audit(
            "security.page_rejected", "deny", "prompt injection in fetched text",
            url=page.url, phrases=verdict.phrases, score=verdict.score,
        )  # fmt: skip
        self._r.telemetry.stage(
            "content_defense",
            f"{node.node_id} · dropped {host_of(page.url)} — phrase [{phrases}], {verdict.why}",
            detail,
            status="blocked",
        )
        return False

    def _admit(
        self,
        node: SubQuestion,
        candidates: list[ClaimCandidate],
        hit: SearchHit,
        page: FetchedPage,
        result: NodeResult,
    ) -> tuple[list[EvidenceEntry], list[dict[str, str]]]:
        """Append each claim under the verbatim rule; the page's own text is what is stored."""
        admitted: list[EvidenceEntry] = []
        refused: list[dict[str, str]] = []
        for candidate in candidates:
            span = canonical_span(page.text, candidate.span) or candidate.span
            outcome = self._r.ledger.append(
                EvidenceCandidate(
                    claim=candidate.claim,
                    span=span,
                    source_url=candidate.url,
                    source_hash=candidate.page_hash,
                    retrieved_at=candidate.retrieved_at,
                    source_date=hit.published_date,
                    subject=node.node_id,
                )
            )
            if isinstance(outcome, EvidenceEntry):
                result.entries.append(outcome.entry_id)
                admitted.append(outcome)
                continue
            self._r.trace.span_rejections += 1
            near = nearest_match(page.text, candidate.span)
            refused.append(
                {
                    "reason": outcome.reason_code,
                    "span": candidate.span[:80],
                    "nearest": near.describe(),
                    "page_says": near.context,
                }
            )
        return admitted, refused

    def _page_card(
        self,
        node: SubQuestion,
        page: FetchedPage,
        extracted: int,
        admitted: list[EvidenceEntry],
        refused: list[dict[str, str]],
    ) -> None:
        """One card per page: the site, the evidence kept, and the claims refused and why."""
        tui = get_agent_settings().tui
        heading, sections = cards.page_card(
            node,
            page.url,
            len(page.text),
            page.via == "reader",
            extracted,
            admitted,
            refused,
            tui.reasons,
            tui.clip_chars,
        )
        self._r.telemetry.card(
            "page_result", heading, sections, status="completed" if admitted else "warning"
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

    def _failed(self, stage: str, target: str, exc: Exception, node: str = "") -> None:
        detail = describe_error(exc)
        self._r.trace.errors.append(f"{target}: {detail}")
        self._r.telemetry.audit(stage.replace("worker_", "web."), "deny", detail, target=target)
        self._r.telemetry.stage(
            stage,
            f"{node + ' · ' if node else ''}{target} failed — {detail}",
            {"node": node, "target": target, "error": detail},
            status="failed",
        )


def _fetch_outcome(status: int) -> tuple[str, str]:
    """A failed fetch's reason code and wording: blocked is not dead, and neither is an error."""
    if status in (401, 402, 403, 407, 429, 451):
        return "blocked", "blocked or paywalled"
    if status in (404, 410):
        return "dead_link", "dead link"
    if 200 <= status < 300:
        return "empty", "no readable text"
    return "error", "fetch error"


def _window_label(request: SearchRequest) -> str:
    window = f"{request.recent_days}d" if request.recent_days else "any time"
    return f"{request.topic}, {window}" if request.topic else window


def _quota_spent(exc: BaseException | None) -> bool:
    """Whether a search failed because every provider is out of monthly quota."""
    while exc is not None:
        if isinstance(exc, QuotaExhaustedError):
            return True
        exc = exc.__cause__
    return False
