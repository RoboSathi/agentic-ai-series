"""The report's hallucination guard: assess each cited sentence, rewrite once, remove the rest."""

import json
import re

import pytest

from agentic_commons.evals.entailment import LexicalEntailmentModel
from agentic_commons.evidence import EvidenceEntry, LedgerSnapshot
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.llm.client import LLMClient
from research_agent.domain.report import Report, ReportSection
from research_agent.prompts.base import payload_of, role_of
from research_agent.structured import StructuredCaller
from research_agent.synthesis.citation_guard import CitationGuard
from tests.conftest import ScriptedLLM

pytestmark = pytest.mark.usefixtures("ra_config")

PATTERN = re.compile(r"\[(E\d+)\]")
LEDGER = LedgerSnapshot(
    entries=(
        EvidenceEntry(
            entry_id="E1",
            claim="The course lasts four years.",
            span="The B.Tech course lasts four years.",
            source_url="https://a.example/p",
            source_hash="h",
        ),
        EvidenceEntry(
            entry_id="E2",
            claim="A MacBook is included in the fee.",
            span="An Apple MacBook is included in the above course fee.",
            source_url="https://a.example/p",
            source_hash="h",
        ),
        EvidenceEntry(
            entry_id="E3",
            claim="Seed fund for startups.",
            span="Upto 1Cr seed fund for startups.",
            source_url="https://b.example/v",
            source_hash="h",
        ),
    )
)
BODY = (
    "- The B.Tech course lasts four years [E1].\n"
    "- Every student gets a personal gaming laptop worth two lakh [E2].\n"
    "- Graduates receive a guaranteed job offer from Google [E3]."
)


class _Roles:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def client_for(self, role: str) -> LLMClient:
        return self.llm

    def model_key(self, role: str) -> str:
        return role


def _guard(rewrites: list[dict[str, object]]) -> tuple[CitationGuard, ScriptedLLM]:
    llm = ScriptedLLM([{"content": json.dumps({"rewrites": rewrites})}])
    caller = StructuredCaller(_Roles(llm), PydanticValidator())
    guard = CitationGuard(LexicalEntailmentModel(), caller, PATTERN, frozenset({"Sources"}), 1)
    return guard, llm


def _report() -> Report:
    return Report(
        title="t",
        status="PARTIAL",
        sections=[
            ReportSection(heading="Answer", body=BODY),
            ReportSection(heading="Sources", body="- left alone [E9]"),
        ],
    )


@pytest.mark.asyncio
async def test_unsupported_sentences_are_rewritten_from_their_spans_or_removed() -> None:
    guard, llm = _guard(
        [
            {"index": 0, "text": "An Apple MacBook is included in the course fee [E2]."},
            {"index": 1, "text": ""},
        ]
    )
    guarded = await guard.guard(_report(), LEDGER)
    body = guarded.report.sections[0].body
    assert (guarded.checked, guarded.supported) == (3, 1)
    assert guarded.rewritten == ["An Apple MacBook is included in the course fee [E2]."]
    assert guarded.dropped == ["Graduates receive a guaranteed job offer from Google [E3]."]
    assert body == (
        "- The B.Tech course lasts four years [E1].\n"
        "- An Apple MacBook is included in the course fee [E2]."
    )
    assert guarded.report.sections[1].body == "- left alone [E9]"  # code-written: not checked
    # The rewrite is the quarantined extractor's, and it is shown the spans, not the pages.
    request = llm.requests[0].messages
    assert role_of(request) == "extractor"
    assert payload_of(request)["items"][0]["spans"] == [LEDGER.entries[1].span]


@pytest.mark.asyncio
async def test_a_rewrite_citing_new_evidence_is_not_accepted() -> None:
    guard, _ = _guard([{"index": 0, "text": "A laptop is included [E1][E2]."}])
    report = Report(
        title="t",
        status="PARTIAL",
        sections=[ReportSection(heading="Answer", body="Each student gets a gaming laptop [E2].")],
    )
    guarded = await guard.guard(report, LEDGER)
    assert guarded.dropped == ["Each student gets a gaming laptop [E2]."]
    assert guarded.report.sections[0].body == ""


@pytest.mark.asyncio
async def test_a_supported_report_is_left_exactly_as_written() -> None:
    guard, llm = _guard([])
    report = Report(
        title="t",
        status="PARTIAL",
        sections=[ReportSection(heading="Answer", body="The B.Tech course lasts four years [E1].")],
    )
    guarded = await guard.guard(report, LEDGER)
    assert guarded.report == report and not guarded.dropped and not llm.requests
