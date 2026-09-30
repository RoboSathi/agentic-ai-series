"""The synthesizer: a report drafted from the ledger, checked, and completed in code.

The model writes the archetype's sections from ledger claims (never
pages). Code then removes any sentence citing an id the ledger did not
issue, attaches the citations (URL, span, page hash, retrieved-at — so the
report stays auditable after the ledger is discarded), and appends the
sections a model must not write: Known unknowns and Sources. Coverage &
Limits is attached by `with_limits`, from verdicts.
"""

import re
from pathlib import Path
from typing import Any

import yaml

from agentic_commons.evidence import LedgerSnapshot
from research_agent.domain.report import (
    COVERAGE_AND_LIMITS,
    Citation,
    LimitEntry,
    Report,
    ReportSection,
    RunStatus,
)
from research_agent.domain.rubric import Contract
from research_agent.prompts import synthesizer as prompt
from research_agent.research.resources import RunTrace
from research_agent.structured import StructuredCaller
from research_agent.synthesis.citation_check import CitationCheck, cited_ids, enforce_citations
from research_agent.synthesis.limits import render_limits


class Synthesizer:
    """Draft → citation check → attach citations and appended sections."""

    def __init__(
        self, caller: StructuredCaller, templates_path: Path, citation_pattern: str
    ) -> None:
        self._caller = caller
        self._templates: dict[str, Any] = yaml.safe_load(templates_path.read_text())
        self._pattern = re.compile(citation_pattern)

    @property
    def pattern(self) -> re.Pattern[str]:
        return self._pattern

    async def draft(
        self, contract: Contract, ledger: LedgerSnapshot, gaps: list[str]
    ) -> CitationCheck:
        headings = list(self._templates[contract.brief.archetype]["sections"])
        answer = await self._caller.call(
            prompt.build(contract, ledger, headings, gaps), prompt.SynthesisAnswer
        )
        report = Report(
            title=answer.title,
            status="PARTIAL",
            sections=[ReportSection(heading=s.heading, body=s.body) for s in answer.sections],
            known_unknowns=[unknown.question for unknown in contract.brief.known_unknowns],
        )
        return enforce_citations(report, ledger, self._pattern)

    def with_limits(
        self,
        report: Report,
        ledger: LedgerSnapshot,
        limits: list[LimitEntry],
        trace: RunTrace,
        status: RunStatus,
    ) -> Report:
        """The deliverable: model sections, then the sections only code writes."""
        appended = self._templates["appended"]
        body = [s for s in report.sections if s.heading not in _appended_headings(appended)]
        sections = [
            *body,
            ReportSection(heading=appended["known_unknowns"], body=_bullets(report.known_unknowns)),
            render_limits(limits, trace.rejected_sources, trace.budget_stops),
        ]
        final = report.model_copy(
            update={
                "sections": sections,
                "limits": limits,
                "status": status,
                "rejected_sources": sorted(trace.rejected_sources),
            }
        )
        return final.model_copy(update={"citations": self.citations(final, ledger)})

    def citations(self, report: Report, ledger: LedgerSnapshot) -> list[Citation]:
        entries = [ledger.entry(entry_id) for entry_id in cited_ids(report, self._pattern)]
        return [
            Citation(
                entry_id=e.entry_id,
                url=e.source_url,
                span=e.span,
                page_hash=e.source_hash,
                retrieved_at=e.retrieved_at,
            )
            for e in entries
            if e is not None
        ]


def _appended_headings(appended: dict[str, str]) -> set[str]:
    return {*appended.values(), COVERAGE_AND_LIMITS}


def _bullets(items: list[str]) -> str:
    return (
        "\n".join(f"- {item}" for item in items)
        if items
        else "- None: every scope question was settled."
    )
