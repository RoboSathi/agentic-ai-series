"""The synthesizer: a report drafted from the ledger, checked, and completed in code.

The model writes the archetype's sections from ledger claims (never
pages). Code then removes any sentence citing an id the ledger did not
issue, attaches the citations (URL, span, page hash, retrieved-at — so the
report stays auditable after the ledger is discarded), and appends the
sections a model must not write: Known unknowns and Sources. Coverage &
Limits is attached by `with_limits`, from verdicts.
"""

import re
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from agent_platform.evidence import EvidenceEntry, LedgerSnapshot
from agent_platform.logging import instrument
from agent_platform.observability.step_card import ModelCall
from research_agent.config.settings import DepthProfile
from research_agent.domain.report import (
    COVERAGE_AND_LIMITS,
    Citation,
    LimitEntry,
    Report,
    ReportSection,
    RunStatus,
)
from research_agent.domain.rubric import Contract
from research_agent.domain.scope import Depth
from research_agent.domain.trace import RunTrace
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import synthesizer as prompt
from research_agent.synthesis.citation_check import (
    CitationCheck,
    cited_ids,
    enforce_citations,
    normalize_citations,
    strip_working_notes,
)
from research_agent.synthesis.limits import render_limits


@instrument("research_agent")
class Synthesizer:
    """Draft → citation check → attach citations and appended sections."""

    def __init__(
        self,
        caller: StructuredCaller,
        templates_path: Path,
        citation_pattern: str,
        depths: Mapping[Depth, DepthProfile],
        working_notes: Sequence[str] = (),
    ) -> None:
        self._caller = caller
        self._notes = [re.compile(pattern, re.IGNORECASE) for pattern in working_notes]
        self._depths = depths
        self._templates: dict[str, Any] = yaml.safe_load(templates_path.read_text())
        self._pattern = re.compile(citation_pattern)

    @property
    def pattern(self) -> re.Pattern[str]:
        return self._pattern

    async def draft(
        self,
        contract: Contract,
        ledger: LedgerSnapshot,
        gaps: list[str],
        previous: Report | None = None,
        findings: list[dict[str, Any]] | None = None,
    ) -> CitationCheck:
        """The report from the ledger; after a gap-fill, a revision of `previous` that keeps
        what the audit accepted and fixes what its `findings` flagged, with the new evidence."""
        headings = list(self._templates[contract.brief.archetype]["sections"])
        if not ledger.entries:
            # Nothing was established: a model asked to write from no evidence can only
            # produce filler. The Coverage & Limits section says why, in code.
            empty = Report(
                title=contract.brief.topic,
                status="PARTIAL",
                sections=[ReportSection(heading=heading, body="") for heading in headings],
                known_unknowns=[unknown.question for unknown in contract.brief.known_unknowns],
            )
            return CitationCheck(report=empty, removed=[], unknown_ids=[])
        size = self._depths[contract.brief.depth]
        shown = _spread(ledger, size.report_entries)
        sections = self.written_sections(previous) if previous is not None else None
        answer, call = await self._caller.call_traced(
            prompt.build(contract, shown, headings, gaps, sections, findings),
            prompt.SynthesisAnswer,
            max_tokens=size.report_max_tokens,
        )
        return self._checked(answer, call, contract, ledger)

    async def revise(
        self,
        contract: Contract,
        ledger: LedgerSnapshot,
        previous: Report,
        findings: list[dict[str, Any]],
        learned_skills: Sequence[str] = (),
    ) -> CitationCheck:
        """The report revised against the audit's findings: only what they flag changes."""
        headings = list(self._templates[contract.brief.archetype]["sections"])
        size = self._depths[contract.brief.depth]
        answer, call = await self._caller.call_traced(
            prompt.build_revision(
                contract,
                _spread(ledger, size.report_entries),
                headings,
                self.written_sections(previous),
                findings,
                learned_skills,
            ),
            prompt.SynthesisAnswer,
            max_tokens=size.report_max_tokens,
        )
        return self._checked(answer, call, contract, ledger)

    def written_sections(self, report: Report) -> list[dict[str, str]]:
        """The sections the model wrote, without those code appends (limits, unknowns)."""
        appended = _appended_headings(self._templates["appended"])
        return [
            {"heading": s.heading, "body": s.body}
            for s in report.sections
            if s.heading not in appended and s.body.strip()
        ]

    def _checked(
        self,
        answer: prompt.SynthesisAnswer,
        call: ModelCall | None,
        contract: Contract,
        ledger: LedgerSnapshot,
    ) -> CitationCheck:
        report = Report(
            title=answer.title,
            status="PARTIAL",
            sections=[
                ReportSection(heading=s.heading, body=normalize_citations(s.body, self._pattern))
                for s in answer.sections
            ],
            known_unknowns=[unknown.question for unknown in contract.brief.known_unknowns],
        )
        checked = enforce_citations(report, ledger, self._pattern)
        cleaned = strip_working_notes(checked, self._notes, self._pattern)
        return cleaned.model_copy(update={"call": call})

    def with_limits(
        self,
        report: Report,
        ledger: LedgerSnapshot,
        limits: list[LimitEntry],
        trace: RunTrace,
        status: RunStatus,
        notice: str = "",
    ) -> Report:
        """The deliverable: model sections, then the sections only code writes.

        `notice` is said above the findings (a best-effort report's warning).
        """
        appended = self._templates["appended"]
        body = [
            s
            for s in report.sections
            if s.heading not in _appended_headings(appended) and s.body.strip()
        ]
        sections = [
            *body,
            ReportSection(heading=appended["known_unknowns"], body=_bullets(report.known_unknowns)),
            render_limits(
                limits,
                trace.rejected_sources,
                trace.budget_stops,
                trace.rejected_codes,
                self._weaker(report, ledger),
            ),
        ]
        final = report.model_copy(
            update={
                "notice": notice,
                "sections": sections,
                "limits": limits,
                "status": status,
                "rejected_sources": sorted(trace.rejected_sources),
                "rejected_codes": dict(trace.rejected_codes),
                "budget_notes": list(trace.budget_stops),
            }
        )
        return final.model_copy(update={"citations": self.citations(final, ledger)})

    def _weaker(self, report: Report, ledger: LedgerSnapshot) -> list[str]:
        """One limits line per caveat that cited findings rest on, with how many."""
        cited = [ledger.entry(entry_id) for entry_id in cited_ids(report, self._pattern)]
        counts = Counter(entry.caveat for entry in cited if entry is not None and entry.caveat)
        line = self._templates["weaker_evidence"]
        return [line.format(count=count, caveat=caveat) for caveat, count in counts.items()]

    def citations(self, report: Report, ledger: LedgerSnapshot) -> list[Citation]:
        entries = [ledger.entry(entry_id) for entry_id in cited_ids(report, self._pattern)]
        return [
            Citation(
                entry_id=e.entry_id,
                url=e.source_url,
                span=e.span,
                page_hash=e.source_hash,
                retrieved_at=e.retrieved_at,
                caveat=e.caveat,
            )
            for e in entries
            if e is not None
        ]


def _appended_headings(appended: dict[str, str]) -> set[str]:
    return {*appended.values(), COVERAGE_AND_LIMITS}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None declared during scoping."


def _spread(ledger: LedgerSnapshot, limit: int) -> LedgerSnapshot:
    """At most `limit` entries, taken in turn from each sub-question so none is crowded out."""
    queues = [
        deque(ledger.for_subject(subject))
        for subject in dict.fromkeys(entry.subject for entry in ledger.entries)
    ]
    chosen: list[EvidenceEntry] = []
    while len(chosen) < limit and any(queues):
        chosen.extend(queue.popleft() for queue in queues if queue and len(chosen) < limit)
    order = {entry.entry_id: i for i, entry in enumerate(ledger.entries)}
    chosen.sort(key=lambda entry: order[entry.entry_id])
    return ledger.model_copy(update={"entries": tuple(chosen)})
