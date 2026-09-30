"""What every acceptance check is evaluated against.

Exactly the Goldfish's inputs — the contract, the report (absent before
synthesis, when reflection asks "what is still unmet?") and a ledger
snapshot — plus the deterministic helpers the checks share. Nothing else:
no conversation, no pages, no run state.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from agentic_commons.evals.entailment import EntailmentModel
from agentic_commons.evidence import EvidenceEntry, LedgerSnapshot
from agentic_commons.verification.acceptance import CheckOutcome, CheckSpec
from research_agent.domain.report import COVERAGE_AND_LIMITS, Report
from research_agent.domain.rubric import Contract
from research_agent.research.defenses import within_window
from research_agent.research.matching import TermMatcher


class Judge(Protocol):
    """Scores one judge-type item against the report (a model call)."""

    async def judge(
        self, spec: CheckSpec, report: Report, ledger: LedgerSnapshot
    ) -> CheckOutcome: ...


@dataclass(frozen=True)
class CheckContext:
    """The contract, the report (if any), the ledger, and shared helpers."""

    contract: Contract
    ledger: LedgerSnapshot
    report: Report | None
    matcher: TermMatcher
    citation_pattern: re.Pattern[str]
    entity_pattern: re.Pattern[str]
    primary_markers: tuple[str, ...]
    entailment: EntailmentModel | None = None
    judge: Judge | None = None
    # Sections code appends (Known unknowns, Coverage & Limits): not claims.
    appended_headings: frozenset[str] = frozenset({COVERAGE_AND_LIMITS})

    # -- the evidence a check may count -----------------------------------------------

    def evidence(self) -> list[EvidenceEntry]:
        """Before synthesis, the whole ledger; after, only what the report cites."""
        if self.report is None:
            return list(self.ledger.entries)
        cited = set(self.cited_ids())
        return [entry for entry in self.ledger.entries if entry.entry_id in cited]

    def cited_ids(self) -> list[str]:
        if self.report is None:
            return []
        text = "\n".join(section.body for section in self.report.sections)
        return list(dict.fromkeys(self.citation_pattern.findall(text)))

    def about(self, *phrases: str) -> list[EvidenceEntry]:
        """Evidence whose claim mentions every phrase given."""
        return [
            entry for entry in self.evidence() if self.matcher.mentions_all(entry.claim, phrases)
        ]

    def body_sentences(self) -> list[str]:
        """The report's factual sentences (Coverage & Limits and headings excluded)."""
        if self.report is None:
            return []
        bodies = [s.body for s in self.report.sections if s.heading not in self.appended_headings]
        return [line.strip() for body in bodies for line in sentences(body) if line.strip()]

    def within(self, source_date: str, days: int) -> bool:
        """Whether a source date falls inside the window ending at contract approval."""
        return within_window(source_date, self.contract.frozen_at.date(), days)


def host(url: str) -> str:
    return (urlparse(url).hostname or "").casefold()


def distinct_hosts(entries: Iterable[EvidenceEntry]) -> set[str]:
    return {host(entry.source_url) for entry in entries}


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    """Sentences and lines (a table row or bullet is one claim-bearing line)."""
    return [part.lstrip("-* |").strip() for part in _SENTENCE_END.split(text) if part.strip()]
