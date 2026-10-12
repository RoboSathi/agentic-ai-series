"""Every research prompt carries today's date, and "the last N years" is resolved in code.

Regression: run-4f316d864e on 2026-10-04 planned "the last 20 years (2003-2023)" — the planner
had no date and counted back from its training-era "now", and the window sat in the Brief's
`geography` field where nothing turned it into dates.
"""

from datetime import UTC, date, datetime

import pytest

from agent_platform.hallucination_check.evidence import LedgerSnapshot
from research_agent.config import get_agent_settings
from research_agent.domain.report import Report
from research_agent.domain.rubric import Contract
from research_agent.domain.scope import CoverageItem, ScopeBrief
from research_agent.prompts import extractor, interviewer, judge, planner, researcher, synthesizer
from research_agent.prompts.base import TODAY_KEY, TODAY_RULE
from research_agent.research.recency import RecencyGuard
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.window import TimeWindowParser

pytestmark = pytest.mark.usefixtures("ra_config")

TODAY = date(2026, 10, 4)


def _builder() -> BriefBuilder:
    agent = get_agent_settings()
    return BriefBuilder(agent.scoping, agent.budget, agent.depths)


def _boundary(value: str) -> CoverageItem:
    return CoverageItem(
        item_id="scope_boundary", goal="g", brief_field="geography", status="covered", value=value
    )


def test_a_window_in_any_answer_becomes_a_period_on_the_calendar() -> None:
    brief = _builder().build(
        "landscape_brief", "India's Cricket World Cup performance",
        [_boundary("last 20 years")], [], today=TODAY,
    )  # fmt: skip
    assert (brief.period_start, brief.period_end) == (date(2006, 10, 4), TODAY)
    assert brief.recency_days == (TODAY - date(2006, 10, 4)).days


def test_a_window_in_the_topic_counts_too() -> None:
    brief = _builder().build("general", "AI news in the last 3 months", [], [], today=TODAY)
    assert brief.period_start == date(2026, 7, 4)


def test_no_window_means_no_period() -> None:
    brief = _builder().build("general", "how do batteries work", [], [], today=TODAY)
    assert brief.period_start is None and brief.recency_days is None


@pytest.mark.parametrize(
    ("today", "span", "start"),
    [
        (date(2024, 2, 29), (1, "year"), date(2023, 2, 28)),  # no 29 Feb in 2023
        (date(2026, 3, 31), (1, "month"), date(2026, 2, 28)),
        (date(2026, 10, 4), (2, "week"), date(2026, 9, 20)),
    ],
)
def test_period_starts_count_months_and_years_on_the_calendar(
    today: date, span: tuple[int, str], start: date
) -> None:
    parser = TimeWindowParser(get_agent_settings().scoping.time_window)
    assert parser.start(today, span) == start


def _contract() -> Contract:
    brief = ScopeBrief(archetype="general", topic="t", budget=get_agent_settings().budget)
    contract = Contract.freeze("c1", brief, brief_rubric(brief), brief.budget)
    return contract.model_copy(update={"frozen_at": datetime(2026, 10, 4, 9, 0, tzinfo=UTC)})


def brief_rubric(brief: ScopeBrief) -> object:
    from research_agent.rubric.generator import RubricGenerator

    return RubricGenerator(get_agent_settings().rubric.resolved_templates_path).generate(brief)


def test_every_research_prompt_carries_today_and_the_rule() -> None:
    contract, ledger = _contract(), LedgerSnapshot(entries=())
    prompts = [
        planner.build(contract, 4, 2),
        synthesizer.build(contract, ledger, ["Findings"], []),
        judge.build("c", "check", Report(title="t", status="PARTIAL"), ledger, TODAY),
        extractor.build("q", "https://p.example", "page", 3, TODAY),
        researcher.build("q", "https://p.example", "page", 3, {}, TODAY),
        interviewer.build(
            archetype="general",
            request="r",
            topic="t",
            items=[],
            asked_item_id="",
            answer="",
            turn=1,
            today=TODAY,
        ),  # fmt: skip
    ]
    for prompt in prompts:
        assert prompt.payload[TODAY_KEY] == "2026-10-04"
        assert TODAY_RULE in prompt.messages(planner.PlanAnswer)[0].content


def _guard() -> RecencyGuard:
    agent = get_agent_settings()
    return RecencyGuard(agent.recency, TimeWindowParser(agent.scoping.time_window))


def test_a_planned_query_before_the_period_is_noted() -> None:
    guard, start = _guard(), date(2006, 10, 4)
    early = guard.outside_period("India World Cup results 2003 2007 2011", start)
    assert early == (2003,)
    assert "2003" in guard.period_note(early, start, TODAY)
    assert guard.outside_period("India World Cup results 2011 2023", start) == ()
