"""No date named means the latest: a search pinned to a past year, or with only old results.

Regression: asked "India's medal tally in Asian Games" on 2026-10-04, the model searched for
"... 2023 Hangzhou" — the edition it knew from training — and answered about it.
"""

from datetime import date

import pytest

from research_agent.config import get_agent_settings
from research_agent.research.recency import RecencyGuard
from research_agent.scoping.window import TimeWindowParser

pytestmark = pytest.mark.usefixtures("ra_config")

TODAY = date(2026, 10, 4)
QUESTION = "what is india's medal tally in asian games"


def _guard() -> RecencyGuard:
    agent = get_agent_settings()
    return RecencyGuard(agent.recency, TimeWindowParser(agent.scoping.time_window))


def test_a_query_pinned_to_a_past_year_the_question_never_named_is_flagged() -> None:
    guard = _guard()
    verdict = guard.check(QUESTION, "India medal tally Asian Games 2023 Hangzhou", [], TODAY)
    assert verdict.pinned == (2023,)
    note = guard.note(verdict, TODAY, "Sunday, 4 October 2026")
    assert "2023" in note and "2026" in note


def test_a_year_the_user_named_is_theirs_to_ask_for() -> None:
    question = "india's medal tally at the 2023 asian games"
    verdict = _guard().check(question, "India medal tally Asian Games 2023", [], TODAY)
    assert not verdict.flagged


def test_a_time_window_the_user_named_dates_the_question() -> None:
    verdict = _guard().check("AI news in the last 30 days", "AI news 2025", ["2025-01-01"], TODAY)
    assert not verdict.flagged


def test_the_current_year_is_the_latest() -> None:
    assert not _guard().check(QUESTION, "India medal tally Asian Games 2026", [], TODAY).flagged


def test_results_that_are_all_old_get_the_stale_note() -> None:
    guard = _guard()
    verdict = guard.check(QUESTION, "India medal tally Asian Games", ["2023-10-08", ""], TODAY)
    assert verdict.stale and verdict.newest == date(2023, 10, 8)
    assert "2023-10-08" in guard.note(verdict, TODAY, "today")


def test_one_recent_result_is_enough() -> None:
    published = ["2023-10-08", "2026-10-01T09:00:00"]
    assert not _guard().check(QUESTION, "India medal tally", published, TODAY).flagged
