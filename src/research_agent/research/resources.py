"""What one run's workers share: the gateway, the ledger, the budget, the defences.

Held by the run, never by the graph state (none of it is serialisable,
and none of it is the kind of thing a checkpoint should carry); what a
resumed run needs back — spend, ledger, examined sources — is snapshotted
into state at each wave boundary instead.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from agentic_commons.control_loop import BudgetGuard
from agentic_commons.evidence import EvidenceLedger
from agentic_commons.resilience import Bulkhead, ResourcePoolBulkhead
from agentic_commons.web import FetchedPage, WebGateway
from research_agent.config.settings import ResearchSettings
from research_agent.domain.rubric import Contract
from research_agent.research.defenses import InjectionScreen, OutboundGuard
from research_agent.research.extractor import PageReading
from research_agent.research.telemetry import RunTelemetry


class PageReader(Protocol):
    """Reads one page for one question (quarantined or not)."""

    async def read(self, question: str, page: FetchedPage) -> PageReading: ...


class RunTrace(BaseModel):
    """Mutable run bookkeeping that ends up in Coverage & Limits and the audit."""

    queries_by_node: dict[str, list[str]] = Field(default_factory=dict)
    sources_examined: list[str] = Field(default_factory=list)
    rejected_sources: dict[str, str] = Field(default_factory=dict)
    # A short machine code per rejected URL (recency, undated, dead_link, injection),
    # so a report can group rejections instead of listing every one.
    rejected_codes: dict[str, str] = Field(default_factory=dict)
    blocked_outbound: list[str] = Field(default_factory=list)
    attempted_outbound: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    span_rejections: int = 0
    provider_summaries_seen: list[str] = Field(default_factory=list)
    budget_stops: list[str] = Field(default_factory=list)

    def tried(self, node_id: str, query: str) -> None:
        self.queries_by_node.setdefault(node_id, []).append(query)

    def examined(self, url: str) -> None:
        if url not in self.sources_examined:
            self.sources_examined.append(url)

    def reject(self, url: str, reason: str, code: str = "other") -> None:
        self.rejected_sources.setdefault(url, reason)
        self.rejected_codes.setdefault(url, code)


@dataclass
class RunResources:
    """The run-scoped collaborators every worker uses."""

    contract: Contract
    gateway: WebGateway
    ledger: EvidenceLedger
    budget: BudgetGuard
    reader: PageReader
    telemetry: RunTelemetry
    settings: ResearchSettings
    outbound: OutboundGuard | None = None
    screen: InjectionScreen | None = None
    # Bounds how many pages the run reads at once (the platform bulkhead, one pool).
    page_reads: Bulkhead = field(default_factory=ResourcePoolBulkhead)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    trace: RunTrace = field(default_factory=RunTrace)

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()
