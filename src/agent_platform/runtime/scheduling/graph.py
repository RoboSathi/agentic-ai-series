"""A validated dependency graph that answers "what may run now?".

Validation happens once, at construction: a graph that could deadlock (a
cycle), mislead (a duplicate id) or wait forever (a dependency nobody
declared) is refused before any work starts rather than discovered half
way through it.
"""

from collections.abc import Iterator, Mapping, Sequence

from agent_platform.runtime.scheduling.types import InvalidTaskGraphError, OutcomeStatus, TaskNode

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
        """Unfinished nodes whose every dependency is done — the next batch."""
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

    def batches(self) -> list[list[str]]:
        """The static batch decomposition, assuming every node succeeds."""
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

    Depth-first with three colours, iterative over one shared path, so no
    path is copied per step and no plan is deep enough to hit the recursion
    limit. Returned rather than raised so a validator can report *which*
    cycle a planner produced.
    """
    edges = {node.node_id: node.depends_on for node in nodes}
    colour: dict[str, int] = {}
    for start in edges:
        cycle = _visit(start, edges, colour)
        if cycle:
            return cycle
    return None


_GREY, _BLACK = 1, 2


def _visit(
    start: str, edges: Mapping[str, Sequence[str]], colour: dict[str, int]
) -> list[str] | None:
    """Walk from `start`; the path back to a grey node, closed, is the cycle."""
    if colour.get(start) == _BLACK:
        return None
    path: list[str] = [start]
    pending: list[Iterator[str]] = [iter(edges.get(start, ()))]
    colour[start] = _GREY
    while pending:
        dep = next(pending[-1], None)
        if dep is None:
            colour[path.pop()] = _BLACK
            pending.pop()
        elif colour.get(dep) == _GREY:
            return [*path[path.index(dep) :], dep]
        elif colour.get(dep) != _BLACK:
            colour[dep] = _GREY
            path.append(dep)
            pending.append(iter(edges.get(dep, ())))
    return None
