"""A validated dependency graph that answers "what may run now?".

Validation happens once, at construction: a graph that could deadlock (a
cycle), mislead (a duplicate id) or wait forever (a dependency nobody
declared) is refused before any work starts rather than discovered half
way through it.
"""

from collections.abc import Mapping, Sequence

from agent_platform.scheduling.types import InvalidTaskGraphError, OutcomeStatus, TaskNode

# Outcomes that let a dependent run, and outcomes that stop it for good.
_SATISFIED: frozenset[OutcomeStatus] = frozenset({"done"})
_POISONED: frozenset[OutcomeStatus] = frozenset({"failed", "blocked"})


class TaskGraph:
    """Nodes plus dependencies, checked acyclic and closed."""

    def __init__(self, nodes: Sequence[TaskNode]) -> None:
        self._nodes = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise InvalidTaskGraphError("duplicate node ids in the task graph")
        self._check_closed()
        cycle = find_cycle(nodes)
        if cycle:
            raise InvalidTaskGraphError(f"cycle: {' -> '.join(cycle)}", cycle)

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes)

    def node(self, node_id: str) -> TaskNode:
        return self._nodes[node_id]

    def ready(self, finished: Mapping[str, OutcomeStatus]) -> list[str]:
        """Unfinished nodes whose every dependency is done — the next wave."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if node_id not in finished
            and all(finished.get(dep) in _SATISFIED for dep in node.depends_on)
        ]

    def blocked(self, finished: Mapping[str, OutcomeStatus]) -> list[str]:
        """Unfinished nodes that can never run: a dependency failed or is blocked."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if node_id not in finished
            and any(finished.get(dep) in _POISONED for dep in node.depends_on)
        ]

    def waves(self) -> list[list[str]]:
        """The static wave decomposition, assuming every node succeeds."""
        finished: dict[str, OutcomeStatus] = {}
        layers: list[list[str]] = []
        while len(finished) < len(self._nodes):
            layer = self.ready(finished)
            layers.append(layer)
            finished.update({node_id: "done" for node_id in layer})
        return layers

    def _check_closed(self) -> None:
        for node in self._nodes.values():
            unknown = [dep for dep in node.depends_on if dep not in self._nodes]
            if unknown:
                raise InvalidTaskGraphError(
                    f"'{node.node_id}' depends on undeclared node(s): {', '.join(unknown)}"
                )


def find_cycle(nodes: Sequence[TaskNode]) -> list[str] | None:
    """One dependency cycle, as a closed path of ids, or None if acyclic.

    Depth-first with three colours. Returned rather than raised so a
    validator can report *which* cycle a planner produced.
    """
    edges = {node.node_id: node.depends_on for node in nodes}
    colour: dict[str, int] = {}
    for start in edges:
        path = _visit(start, edges, colour, [])
        if path:
            return path
    return None


_GREY, _BLACK = 1, 2


def _visit(
    node_id: str, edges: Mapping[str, Sequence[str]], colour: dict[str, int], stack: list[str]
) -> list[str] | None:
    if colour.get(node_id) == _BLACK:
        return None
    if colour.get(node_id) == _GREY:
        return [*stack[stack.index(node_id) :], node_id]
    colour[node_id] = _GREY
    for dep in edges.get(node_id, ()):
        cycle = _visit(dep, edges, colour, [*stack, node_id])
        if cycle:
            return cycle
    colour[node_id] = _BLACK
    return None
