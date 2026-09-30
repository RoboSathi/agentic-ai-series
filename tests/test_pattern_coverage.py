"""`make patterns`: the catalogue × agent matrix and its gaps."""

from pathlib import Path

import pytest

from design_patterns.catalogue import ALL_PATTERNS
from design_patterns.coverage import AgentPatterns, PatternUse, coverage, load_agent, main, markdown

AGENTS = ["--agent", "csa=customer_support_agent", "--agent", "ra=research_agent"]


def _agent(*patterns: str) -> AgentPatterns:
    return AgentPatterns(
        agent="a",
        patterns=[PatternUse(pattern=p, where=f"{p}.py") for p in patterns],
        other=[PatternUse(pattern="saga", where="saga.py")],
    )


def test_every_catalogue_pattern_gets_a_row_and_unused_ones_are_flagged() -> None:
    result = coverage({"a": _agent("router"), "b": _agent("router", "react")})
    assert [row[0] for row in result.rows] == [card.name for card in ALL_PATTERNS]
    assert "router" not in result.unused and "react" not in result.unused
    assert "sequence" in result.unused and result.other[0] == ["saga", "a", "saga.py"]


def test_a_name_outside_the_catalogue_is_reported() -> None:
    assert coverage({"a": _agent("not_a_pattern")}).unknown == ["a: not_a_pattern"]


def test_both_agents_declare_only_catalogue_patterns(tmp_path: Path) -> None:
    out = tmp_path / "coverage.md"
    assert main([*AGENTS, "--markdown", str(out)]) == 0
    assert "| pattern | family | taught in | csa | ra |" in out.read_text()


def test_the_committed_matrix_is_current() -> None:
    agents = {"csa": load_agent("customer_support_agent"), "ra": load_agent("research_agent")}
    committed = Path(__file__).resolve().parents[1] / "design_docs" / "patterns_coverage.md"
    assert committed.read_text() == markdown(coverage(agents))


def test_an_unknown_pattern_fails_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "design_patterns.coverage.load_agent", lambda package: _agent("not_a_pattern")
    )
    assert main(["--agent", "x=y"]) == 1
