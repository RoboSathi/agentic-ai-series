"""Grouped rejections in Coverage & Limits, and follow-ups that fit the remaining searches."""

from research_agent.domain.plan import SubQuestion
from research_agent.graph.run_nodes import fit_to_budget
from research_agent.synthesis.limits import render_limits


def _rejected(n: int, code: str, why: str) -> tuple[dict[str, str], dict[str, str]]:
    urls = [f"https://example.com/{code}/{i}" for i in range(n)]
    return {u: f"{why} (published day {i})" for i, u in enumerate(urls)}, dict.fromkeys(urls, code)


def test_rejections_are_grouped_by_reason_not_listed_one_by_one() -> None:
    old, old_codes = _rejected(15, "recency", "outside the approved 30-day recency window")
    dead, dead_codes = _rejected(1, "dead_link", "dead link")
    section = render_limits([], {**old, **dead}, [], {**old_codes, **dead_codes})
    lines = [line for line in section.body.splitlines() if "rejected" in line]
    assert len(lines) == 2
    assert lines[0].startswith("- 15 source(s) rejected — outside the approved 30-day recency")
    assert "+13 more" in lines[0]
    assert "published day" not in section.body  # per-URL detail is not repeated


def test_render_limits_without_codes_still_groups_by_reason_text() -> None:
    old, _ = _rejected(3, "x", "outside the approved 30-day recency window")
    body = render_limits([], old, []).body
    assert body.count("source(s) rejected") == 1


def _node(node_id: str, *queries: str) -> SubQuestion:
    return SubQuestion(node_id=node_id, question="q", rubric_items=("R1",), queries=queries)


def test_follow_ups_are_trimmed_to_the_searches_left() -> None:
    nodes = [_node("F2-1", "a"), _node("F2-2", "b")]
    fitted = fit_to_budget(nodes, searches_left=1)
    assert [n.node_id for n in fitted] == ["F2-1"]


def test_a_node_with_more_queries_than_room_is_cut_not_dropped() -> None:
    fitted = fit_to_budget([_node("F2-1", "a", "b", "c")], searches_left=2)
    assert fitted[0].queries == ("a", "b")


def test_no_searches_left_means_no_follow_ups() -> None:
    assert fit_to_budget([_node("F2-1", "a")], searches_left=0) == []
