"""What one run's workers share: the gateway, the ledger, the budget, the defences.

Held by the run, never by the graph state (none of it is serialisable,
and none of it is the kind of thing a checkpoint should carry); what a
resumed run needs back — spend, ledger, examined sources — is snapshotted
into state at each batch boundary instead.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from agent_platform.evidence import EvidenceLedger
from agent_platform.loop_control import BudgetGuard
from agent_platform.resilience import Bulkhead, ResourcePoolBulkhead
from agent_platform.web import FetchedPage, WebGateway
from research_agent.config.settings import ResearchSettings
from research_agent.domain.rubric import Contract
from research_agent.domain.trace import RunTrace
from research_agent.guardrails.injection import InjectionScreen
from research_agent.observability.telemetry import RunTelemetry
from research_agent.research.defenses import OutboundGuard
from research_agent.research.extractor import PageReading
from research_agent.research.recency import RecencyGuard
from research_agent.research.snippets import SnippetSources


class PageReader(Protocol):
    """Reads one page for one question (quarantined or not)."""

    async def read(self, question: str, page: FetchedPage) -> PageReading: ...


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
    # No date named means the latest: a planned query pinned to the past is shown, not rewritten.
    recency: RecencyGuard | None = None
    # The verbatim rule's texts: fetched pages, plus the snippets admitted in a page's place.
    snippets: SnippetSources | None = None
    # Bounds how many pages the run reads at once (the platform bulkhead, one pool).
    page_reads: Bulkhead = field(default_factory=ResourcePoolBulkhead)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    trace: RunTrace = field(default_factory=RunTrace)

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    @property
    def recency_relaxed(self) -> bool:
        """Whether the rescue round has run: out-of-window sources are admitted, labelled."""
        return self.trace.rescued
