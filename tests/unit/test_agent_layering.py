"""Inside each agent, dependencies point inward and never round in a circle.

`domain` is what the business is made of: it imports nothing else from its
own agent (the platform beneath it is fine). Storage depends on the domain,
never the reverse; that is what `domain/ports.py` is for. And no two
subpackages depend on each other, however long the way round: a cycle means
neither can be understood, tested or replaced without the other.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
PLATFORM = "agent_platform"
AGENTS = sorted(
    path.parent.name for path in SRC.glob("*/__init__.py") if path.parent.name != PLATFORM
)
# Subpackages that sit at the bottom of an agent: they may import only themselves.
INNERMOST = ("domain",)
# Subpackages that must never reach up into the agent's outer layers.
NEVER_UPWARD = {"persistence": ("graph", "console")}


def _edges(agent: str) -> dict[str, set[str]]:
    """Subpackage -> the sibling subpackages it imports (lazy imports included)."""
    root = SRC / agent
    edges: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if "__pycache__" in parts or len(parts) < 2:
            continue
        source = parts[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in names:
                pieces = name.split(".")
                if pieces[0] == agent and len(pieces) > 1 and pieces[1] != source:
                    edges.setdefault(source, set()).add(pieces[1])
    return edges


def _cycle(edges: dict[str, set[str]]) -> list[str]:
    """One import cycle among the subpackages, or [] when there is none."""
    state: dict[str, int] = {}
    path: list[str] = []

    def visit(node: str) -> list[str]:
        state[node] = 1
        path.append(node)
        for nxt in sorted(edges.get(node, ())):
            if state.get(nxt) == 1:
                return [*path[path.index(nxt) :], nxt]
            if nxt not in state and (found := visit(nxt)):
                return found
        path.pop()
        state[node] = 2
        return []

    for node in sorted(edges):
        if node not in state and (found := visit(node)):
            return found
    return []


@pytest.mark.parametrize("agent", AGENTS)
def test_the_innermost_layers_import_nothing_of_their_agent(agent: str) -> None:
    edges = _edges(agent)
    offenders = {inner: sorted(edges.get(inner, ())) for inner in INNERMOST if edges.get(inner)}
    assert offenders == {}, f"{agent}: inner layers reach outward: {offenders}"


@pytest.mark.parametrize("agent", AGENTS)
def test_storage_never_reaches_up(agent: str) -> None:
    edges = _edges(agent)
    offenders = {
        lower: sorted(edges.get(lower, set()) & set(upper))
        for lower, upper in NEVER_UPWARD.items()
        if edges.get(lower, set()) & set(upper)
    }
    assert offenders == {}, f"{agent}: {offenders}"


@pytest.mark.parametrize("agent", AGENTS)
def test_no_subpackages_import_each_other_in_a_cycle(agent: str) -> None:
    cycle = _cycle(_edges(agent))
    assert cycle == [], f"{agent}: import cycle {' -> '.join(cycle)}"
