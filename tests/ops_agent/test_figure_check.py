"""The figures check: every number in an answer must be one a query returned."""

import pytest

from ops_agent.config import get_agent_settings
from ops_agent.domain import QueryRecord
from ops_agent.guardrails import FigureCheck


@pytest.fixture
def check(ops_config: None) -> FigureCheck:
    return FigureCheck(get_agent_settings())


def _query(*rows: list[object]) -> list[QueryRecord]:
    return [QueryRecord(sql="SELECT 1", columns=["a", "b"], rows=list(rows), row_count=len(rows))]


def test_a_figure_from_the_results_is_supported(check: FigureCheck) -> None:
    assert check.unsupported("p95 was 1,234.5 ms", _query(["m", 1234.5])) == []


def test_a_rounded_figure_is_supported(check: FigureCheck) -> None:
    assert check.unsupported("about 1235 ms", _query(["m", 1234.567])) == []


def test_a_percentage_of_a_ratio_is_supported(check: FigureCheck) -> None:
    assert check.unsupported("a 95.4% success rate", _query(["m", 0.954])) == []


def test_an_invented_figure_is_not(check: FigureCheck) -> None:
    assert check.unsupported("there were 4217 calls", _query(["m", 12])) == ["4217"]


def test_dates_are_not_figures(check: FigureCheck) -> None:
    assert check.unsupported("on 2026-10-07 at 14:00", _query(["m", 12])) == []


def test_digits_inside_a_returned_name_are_supported(check: FigureCheck) -> None:
    rows = _query(["nvidia/nemotron-3-ultra-550b", 3])
    assert check.unsupported("nemotron-3-ultra-550b answered", rows) == []


def test_small_counts_in_prose_need_no_query(check: FigureCheck) -> None:
    assert check.unsupported("the top 5 actions", []) == []


def test_an_ordinal_is_not_a_figure(check: FigureCheck) -> None:
    assert check.unsupported("the 95th percentile", []) == []


def test_thousands_grouped_by_spaces_or_commas_are_one_figure(check: FigureCheck) -> None:
    rows = _query(["m", 121322.43])
    assert check.unsupported("about 121 322 ms", rows) == []
    assert check.unsupported("about 121,322 ms", rows) == []
