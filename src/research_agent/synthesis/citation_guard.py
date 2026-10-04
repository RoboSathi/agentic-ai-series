"""The report's hallucination guard: every cited sentence must be entailed by the spans it cites.

The support agent's response guard, in the shape a cited report needs:

    assess    each cited sentence against its spans (NLI, the audit's own rule)
    repair    one bounded rewrite of only the sentences that fail, from their spans
    fail      closed: a sentence still unsupported after the rewrite is removed

It runs before the audit, so the audit grades a report whose every sentence was checked rather
than discovering, after the fact, that a paraphrase said more than its source ("Upto 1Cr seed
fund for startups" written as "₹1 crore seed fund for student startups").

The rewrite is the quarantined extractor's — the only role allowed to read page text. The
writer never sees a span, and nothing the rewrite returns can cite an entry the sentence did
not already cite.
"""

import re

from pydantic import BaseModel, Field

from agent_platform.evals.entailment import EntailmentModel
from agent_platform.evidence import LedgerSnapshot
from agent_platform.logging import instrument
from agent_platform.observability.step_card import ModelCall
from research_agent.audit.checks import sentence_entailed
from research_agent.audit.context import sentences
from research_agent.domain.report import Report, ReportSection
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import extractor as prompt

# A line left with nothing but a list marker once its sentence was removed.
_BARE_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*$")
_SPACES = re.compile(r"[ \t]{2,}")


SUPPORTED, REWRITTEN, DROPPED = "supported", "rewritten", "dropped"


class SentenceCheck(BaseModel):
    """One cited sentence: what became of it, the entries it cites, and its rewrite."""

    sentence: str
    outcome: str = SUPPORTED
    cited: list[str] = Field(default_factory=list)
    rewrite: str = ""


class GuardedReport(BaseModel):
    """The report as the audit will read it, and what the guard did to it."""

    report: Report
    checked: int = 0
    supported: int = 0
    rewritten: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    # Every cited sentence, in report order, with the entries behind it.
    checks: list[SentenceCheck] = Field(default_factory=list)
    # The rewrite call(s), when any sentence needed one.
    call: ModelCall | None = None


@instrument("research_agent")
class CitationGuard:
    """Assess, rewrite once, then remove what is still unsupported."""

    def __init__(
        self,
        entailment: EntailmentModel,
        caller: StructuredCaller,
        citation_pattern: re.Pattern[str],
        appended_headings: frozenset[str],
        max_rewrites: int,
    ) -> None:
        self._entailment = entailment
        self._caller = caller
        self._pattern = citation_pattern
        self._appended = appended_headings
        self._max_rewrites = max_rewrites

    async def guard(self, report: Report, ledger: LedgerSnapshot) -> GuardedReport:
        cited = self._cited(report)
        failing = await self._failing(cited, ledger)
        supported = len(cited) - len(failing)
        origin = {sentence: sentence for sentence in cited}
        checks = {s: SentenceCheck(sentence=s, cited=self._ids(s)) for s in cited}
        rewritten: list[str] = []
        dropped: list[str] = []
        calls: list[ModelCall] = []
        for _ in range(self._max_rewrites):
            if not failing:
                break
            replacements, call = await self._rewrite(failing, ledger)
            calls.append(call)
            report = _replaced(report, replacements, self._appended)
            for old, new in replacements.items():
                first = checks[origin[old]]
                first.outcome, first.rewrite = (REWRITTEN, new) if new else (DROPPED, "")
                if new:
                    origin[new] = origin[old]
            dropped.extend(old for old, new in replacements.items() if not new)
            rewritten.extend(new for new in replacements.values() if new)
            failing = await self._failing([new for new in replacements.values() if new], ledger)
        for sentence in failing:
            checks[origin.get(sentence, sentence)].outcome = DROPPED
        report = _replaced(report, dict.fromkeys(failing, ""), self._appended)
        return GuardedReport(
            report=report,
            checked=len(cited),
            supported=supported,
            rewritten=[s for s in rewritten if s not in failing],
            dropped=[*dropped, *failing],
            checks=list(checks.values()),
            call=ModelCall.combined(calls),
        )

    def _ids(self, sentence: str) -> list[str]:
        """The entry ids a sentence cites, in order, once each."""
        return list(dict.fromkeys(self._pattern.findall(sentence)))

    def _cited(self, report: Report) -> list[str]:
        bodies = [s.body for s in report.sections if s.heading not in self._appended]
        return [s for body in bodies for s in sentences(body) if self._pattern.search(s)]

    async def _failing(self, cited: list[str], ledger: LedgerSnapshot) -> list[str]:
        return [
            s
            for s in cited
            if s and not await sentence_entailed(s, ledger, self._entailment, self._pattern)
        ]

    async def _rewrite(
        self, failing: list[str], ledger: LedgerSnapshot
    ) -> tuple[dict[str, str], ModelCall]:
        """Each failing sentence -> its rewrite ("" when none could be written faithfully)."""
        items: list[dict[str, object]] = []
        for index, sentence in enumerate(failing):
            ids = list(dict.fromkeys(self._pattern.findall(sentence)))
            markers = list(dict.fromkeys(m.group(0) for m in self._pattern.finditer(sentence)))
            spans = [e.span for i in ids if (e := ledger.entry(i)) is not None]
            items.append(
                {"index": index, "sentence": sentence, "spans": spans, "citations": markers}
            )
        answer, call = await self._caller.call_traced(
            prompt.build_rewrite(items), prompt.RewriteAnswer
        )
        by_index = {rewrite.index: rewrite.text.strip() for rewrite in answer.rewrites}
        return {
            sentence: self._faithful(sentence, by_index.get(index, sentence))
            for index, sentence in enumerate(failing)
        }, call

    def _faithful(self, original: str, rewrite: str) -> str:
        """The rewrite, unless it cites what the original did not (then it is not a rewrite)."""
        allowed = set(self._pattern.findall(original))
        cited = set(self._pattern.findall(rewrite))
        if not rewrite or not cited or not cited <= allowed:
            return "" if not rewrite else original
        return rewrite


def _replaced(report: Report, replacements: dict[str, str], appended: frozenset[str]) -> Report:
    """The report with each sentence replaced (or, for "", removed), lists kept tidy."""
    if not replacements:
        return report
    sections = []
    for section in report.sections:
        body = section.body
        if section.heading not in appended:
            for old, new in replacements.items():
                body = body.replace(old, new)
            lines = [_SPACES.sub(" ", line).rstrip() for line in body.splitlines()]
            body = "\n".join(line for line in lines if not _BARE_MARKER.match(line)).strip()
        sections.append(ReportSection(heading=section.heading, body=body))
    return report.model_copy(update={"sections": sections})


__all__ = ["CitationGuard", "GuardedReport"]
