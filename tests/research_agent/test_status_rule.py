"""PARTIAL means unmet or cut off mid-way — not 'a cap was reached' or 'a page was refused'."""

import pytest

from research_agent.graph.run_nodes import truncated_by_budget


@pytest.mark.parametrize(
    ("exhausted", "truncated"),
    [
        (["fetches"], False),  # the last page read committed; nothing was cut off
        (["searches", "rounds"], False),  # the plan ran its course
        (["tokens"], True),  # the model budget cuts a run mid-way
        (["seconds"], True),  # so does the clock
        ([], False),
    ],
)
def test_truncated_by_budget(exhausted: list[str], truncated: bool) -> None:
    assert truncated_by_budget(exhausted) is truncated
