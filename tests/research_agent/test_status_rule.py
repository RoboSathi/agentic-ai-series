"""PARTIAL means unmet or actually cut short — not merely 'a cap was reached'."""

import pytest

from research_agent.graph.run_nodes import truncated_by_budget


@pytest.mark.parametrize(
    ("exhausted", "stops", "truncated"),
    [
        (["fetches"], [], False),  # the 8th fetch committed; nothing was refused
        (["searches", "rounds"], [], False),
        ([], ["n2: fetches exhausted"], True),  # a reservation was refused
        (["tokens"], [], True),  # the model budget cuts a run mid-way
        (["seconds"], [], True),  # so does the clock
        ([], [], False),
    ],
)
def test_truncated_by_budget(exhausted: list[str], stops: list[str], truncated: bool) -> None:
    assert truncated_by_budget(exhausted, stops) is truncated
